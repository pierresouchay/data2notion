#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os
import random
import sys
import time
from abc import abstractmethod
from enum import Enum
from types import TracebackType
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Coroutine,
    Iterable,
    Optional,
    Sequence,
    Type,
    Union,
)

import asynciolimiter
import click
from notion_client import AsyncClient
from notion_client.errors import APIErrorCode, APIResponseError
from notion_client.helpers import async_iterate_paginated_api

from data2notion.plugins.plugin import Plugin, PluginInstance, SourceRecord
from data2notion.serialization import (
    NotionType,
    convert_value_to_notion,
    get_canonical_value_from_notion,
    notion_type_from_str,
)

logger = logging.getLogger("data2notion")


__version__ = "1.0.8"

__plugin_api_version__ = 1.0


class Statistic:
    def __init__(self, name: str) -> None:
        self.name = name
        self.count = 0
        self.seconds = 0.0
        # runnings handle re-entrant with (since we are in async mode)
        self._runnings: list[float] = []

    def increment(self, duration_seconds: float) -> None:
        self.count += 1
        self.seconds += duration_seconds

    def set(self, count: int, duration_seconds: float) -> None:
        self.count += count
        self.seconds = duration_seconds

    def __enter__(self) -> None:
        self._runnings.append(time.perf_counter())

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        self.increment(time.perf_counter() - self._runnings.pop())

    def __repr__(self) -> str:
        res = f"{self.name:<16}: {self.count:>4}"
        if self.seconds > 0:
            res += f" in {round(self.seconds, 1)}s ({round(self.count / self.seconds, 1):<4}/s)"
        return res


class Statistics:
    def __init__(self) -> None:
        self.load_plugins = Statistic("load_plugins")
        self.notion_records = Statistic("notion_records")
        self.source_records = Statistic("source_records")
        self.records_added = Statistic("record_added")
        self.records_removed = Statistic("records_removed")
        self.records_updated = Statistic("records_updated")
        self.ignored_modifications = Statistic("ignored_modifications")

    def iterate_on_stats(self) -> Iterable[Statistic]:
        for attr in dir(self):
            if not attr.startswith("__"):
                stat = getattr(self, attr)
                if isinstance(stat, Statistic):
                    yield stat

    def display_non_empty_stats(self, prefix: str = " - ") -> str:
        return "\n".join(
            [
                f"{prefix}{stat}"
                for stat in filter(lambda s: s.count > 0, self.iterate_on_stats())
            ]
        )

    def __repr__(self) -> str:
        return "\n".join([f" - {stat}" for stat in self.iterate_on_stats()])


stats = Statistics()


def create_rate_limiter(spec: str) -> asynciolimiter._CommonLimiterMixin:
    split_vals = spec.split(":")
    assert (
        len(split_vals) == 2
    ), f"rate limiter spec: <req_per_sec_float>:<capacity_int>, but was: {spec}"
    rate = float(split_vals[0])
    capacity = int(split_vals[1])
    logger.debug(
        "Using notion rate LeakyBucketLimiter %.2f req/s, capacity=%d", rate, capacity
    )
    _rate_limiter = asynciolimiter.LeakyBucketLimiter(rate=rate, capacity=capacity)
    return _rate_limiter


def default_rate_limit() -> str:
    return os.getenv("NOTION_RATE_LIMIT", "3:100")


_rate_limiter = create_rate_limiter(default_rate_limit())


def rate_limiter() -> asynciolimiter._CommonLimiterMixin:
    return _rate_limiter


def available_plugins() -> Iterable[Plugin]:
    additional_plugins = os.getenv("DATA2NOTION_ADDITIONAL_PLUGINS", "").split(",")
    for plugin_spec in additional_plugins + [
        "plugins.csv_plugin:CSVPlugin",
        "plugins.json_plugin:JSONPlugin",
        "plugins.prometheus_plugin:PrometheusPlugin",
    ]:
        with stats.load_plugins:
            plugin_spec = plugin_spec.strip()
            if not plugin_spec:
                continue
            pkg, clz_name = plugin_spec.split(":")
            clz = __import__(
                pkg,
                globals={"__name__": "data2notion.main"},
                fromlist=[clz_name],
                level=1,
            )
            yield getattr(clz, clz_name)()


def display_plugins() -> None:
    print("Registered Plugins")
    for p in available_plugins():
        disabled_reason = p.is_disabled(__plugin_api_version__)
        if disabled_reason:
            msg = f"[ERR]: {disabled_reason}"
        else:
            msg = "[OK]\t"
        print(
            f" {msg} {p.info.name:<12}\t{p.info.version:<8} feat: {p.info.description:<64} by {p.info.author}"
        )


def find_title(rec: SourceRecord, title_in_notion: str) -> str:
    if rec.props.get(title_in_notion):
        return title_in_notion
    raise ValueError(
        f"Don't know how to find Notion title: {title_in_notion} in {list(rec.props.keys())}"
    )


T = str  # the callable/awaitable return type


MAX_HTTP_TRIES = 5


def retry_http() -> (
    Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]
):
    def wrapper(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapped(*args: Any, **kwargs: Any) -> T:
            last_error = ""
            for retry in range(1, MAX_HTTP_TRIES):
                try:
                    await rate_limiter().wait()
                    response = await func(*args, **kwargs)
                    return response
                except APIResponseError as err:
                    if err.code in [
                        APIErrorCode.ConflictError,
                        APIErrorCode.ServiceUnavailable,
                        APIErrorCode.RateLimited,
                    ]:
                        if err.code == APIErrorCode.RateLimited:
                            # https://developers.notion.com/reference/request-limits
                            # integer number of seconds
                            sleep_for = float(err.headers["Retry-After"])
                            assert sleep_for
                        elif err.code == APIErrorCode.ConflictError:
                            # This happens sometimes when doing too many requests at once
                            sleep_for = 1 + retry**2 + random.random() * retry
                        else:
                            sleep_for = retry**2 + random.random() * retry

                        logger.warning(
                            "[RETRY %d] will retry in %ds due to %s: %s",
                            retry,
                            sleep_for,
                            err.code,
                            str(err),
                        )
                        last_error = str(err.code)
                        await asyncio.sleep(sleep_for)
                    else:
                        logger.error(
                            "[ERROR] Notion API Error: %s: %s",
                            err.code,
                            str(err),
                        )
                        raise err
            raise IOError(
                f"Cannot fetch data after {MAX_HTTP_TRIES} tries, last_error={last_error}"
            )

        return wrapped

    return wrapper


class NotionRecord:
    def __init__(self, props: dict[str, Any]):
        self.id = props["id"]
        assert props["archived"] is False
        self._props = props["properties"]

    def get_canonical(self, prop_name: str) -> Any:
        return get_canonical_value_from_notion(self._props.get(prop_name))


def concat_plain_text(notion_structured_text: Iterable[dict[str, Any]]) -> str:
    return "".join(map(lambda a: a.get("plain_text", ""), notion_structured_text))


class NotionDataBaseInfo:
    def __init__(self, retrieve_db_info: dict[str, Any]):
        self.properties: dict[str, str] = {}
        self.title = concat_plain_text(retrieve_db_info.get("title", []))
        self.description = concat_plain_text(retrieve_db_info.get("description", []))
        self.url = retrieve_db_info.get("url", "")
        for k, v in retrieve_db_info.get("properties", {}).items():
            if k:
                typ = v["type"]
                self.properties[k] = typ
                if typ == "title":
                    self.title = k


class ApplyPolicy(str, Enum):
    APPLY = "APPLY"
    CONFIRM = "CONFIRM"
    IGNORE = "IGNORE"


class ApplyPolicies:
    def __init__(self) -> None:
        self.update = ApplyPolicy.APPLY
        self.add = ApplyPolicy.APPLY
        self.remove = ApplyPolicy.APPLY


def truncate_chars(val: str, max_len: int) -> str:
    if len(val) > max_len:
        return val[: max_len - 1] + "…"
    return val


class NotionProcessor:
    def __init__(self, database_id: str, notion_token: str):
        self.notion = AsyncClient(auth=notion_token)
        self.database_id = database_id
        self.db_info = NotionDataBaseInfo({})
        self.apply_policies = ApplyPolicies()

    async def read_db_props(self) -> None:
        db_info = await self.notion.databases.retrieve(database_id=self.database_id)
        self.db_info = NotionDataBaseInfo(db_info)
        print(
            "[START] Syncing",
            self.db_info.url,
            truncate_chars(self.db_info.title, 32),
            f"[{truncate_chars(self.db_info.description, 32)}]…",
            "ver",
            __version__,
        )

    async def iterate_over_pages(self) -> AsyncGenerator[NotionRecord, None]:
        num_notion_records = 0
        async for rec in async_iterate_paginated_api(
            self.notion.databases.query, database_id=self.database_id
        ):
            num_notion_records += 1
            assert isinstance(rec, dict)
            yield NotionRecord(rec)

    def close(self) -> None:
        asyncio.run(self.notion.aclose())


class SourceRecordCanonical:
    def __init__(
        self,
        source_record: SourceRecord,
        plugin: PluginInstance,
        fields_to_dump: dict[str, NotionType],
    ):
        self.source_identifier = source_record.source_identifier
        self.props: dict[str, Any] = {}
        for prop_name, notion_type in fields_to_dump.items():
            val = plugin.get_property_as_canonical_value(
                source_record, prop_name=prop_name, notion_type=notion_type
            )
            if val:
                self.props[prop_name] = val


def parse_source(
    plugin: PluginInstance,
    title_in_notion: str,
    fields_intersection: dict[str, NotionType],
) -> tuple[dict[str, list[SourceRecordCanonical]], int]:
    indexed_source_records: dict[str, list[SourceRecordCanonical]] = {}
    title_in_source = ""
    rows_count = 0
    try:
        plugin.setup_from_notion(
            title_in_notion, fields_intersection=fields_intersection
        )
        for rec_raw in plugin.values():
            rows_count += 1
            if title_in_source == "":
                # Optimization for memory and speed
                props_to_remove = []
                for prop_name, prop_type in fields_intersection.items():
                    if not plugin.supports_property(
                        prop_name,
                        prop_type,
                        first_record=rec_raw,
                    ):
                        props_to_remove.append(prop_name)
                for prop_name in props_to_remove:
                    del fields_intersection[prop_name]
                title_in_source = find_title(
                    rec=rec_raw, title_in_notion=title_in_notion
                ).strip()
                logger.info(
                    "Comparing source/notion fields: %s",
                    ",".join(sorted(fields_intersection.keys())),
                )

            rec = SourceRecordCanonical(
                rec_raw, plugin=plugin, fields_to_dump=fields_intersection
            )
            idx = rec.props.get(title_in_source)
            if idx is None:
                print(
                    f"WARN: could not find {title_in_source} in {rec.source_identifier}"
                )
            else:
                assert isinstance(idx, str), type(idx)
                lst = indexed_source_records.get(idx)
                if lst is None:
                    lst = []
                    indexed_source_records[idx] = lst
                lst.append(rec)
        return (indexed_source_records, rows_count)
    finally:
        plugin.close()


def pop_best_candidate(
    notion_rec: NotionRecord, candidates: list[SourceRecordCanonical]
) -> SourceRecordCanonical:
    if len(candidates) == 1:
        return candidates.pop()
    assert notion_rec
    # TODO: We could optimize this by finding similarities
    return candidates.pop()


class NotionPageModification:
    def __init__(self) -> None:
        pass

    @property
    def modification_type(self) -> str:
        return self.__class__.__name__

    @retry_http()
    @abstractmethod
    async def apply_changes(self, notion_processor: NotionProcessor) -> str:
        assert notion_processor
        raise NotImplementedError()


class PageToRemove(NotionPageModification):
    def __init__(self, page_id: str, title: str):
        self.page_id = page_id
        self.title = title

    def __repr__(self) -> str:
        return f"[DELETE] {self.page_id} {self.title}"

    @retry_http()
    async def apply_changes(self, notion_processor: NotionProcessor) -> str:
        with stats.records_removed:
            await notion_processor.notion.pages.update(
                page_id=self.page_id, archived=True
            )
        return str(self)


class PageToUpdate(NotionPageModification):
    def __init__(self, page_id: str, updates: dict[str, tuple[Any, Any]]):
        self.page_id = page_id
        self.updates = updates

    def __repr__(self) -> str:
        changes = ",".join(
            [f"{kv[0]}=[{kv[1][0]} → {kv[1][1]}]" for kv in self.updates.items()]
        )
        return f"[UPDATE] {self.page_id} {changes}"

    @retry_http()
    async def apply_changes(self, notion_processor: NotionProcessor) -> str:
        with stats.records_updated:
            payload = {}
            for k, old_and_new_val in self.updates.items():
                prop_type_str = notion_processor.db_info.properties[k]
                prop_type = notion_type_from_str(prop_type_str)
                payload[k] = {
                    prop_type.name: convert_value_to_notion(
                        prop_type, old_and_new_val[1]
                    )
                }
            await notion_processor.notion.pages.update(
                page_id=self.page_id, archived=False, properties=payload
            )
        return str(self)


class PageToAdd(NotionPageModification):
    def __init__(self, source_identifier: str, props_to_add: dict[str, str]):
        self.source_identifier = source_identifier
        self.props_to_add = props_to_add

    def __repr__(self) -> str:
        return f"[ADDING] from source id={self.source_identifier}"

    @retry_http()
    async def apply_changes(self, notion_processor: NotionProcessor) -> str:
        with stats.records_added:
            payload = {}
            for k, value in self.props_to_add.items():
                prop_type_str = notion_processor.db_info.properties[k]
                prop_type = notion_type_from_str(prop_type_str)
                payload[k] = {prop_type.name: convert_value_to_notion(prop_type, value)}
            await notion_processor.notion.pages.create(
                parent={"database_id": notion_processor.database_id}, properties=payload
            )
        return str(self)


def find_diffs(
    notion_record: NotionRecord,
    source_record: SourceRecordCanonical,
    fields_to_compare: dict[str, NotionType],
) -> dict[str, tuple[Any, Any]]:
    res: dict[str, tuple[Any, Any]] = {}
    for k, v in fields_to_compare.items():
        assert v
        notion_val = notion_record.get_canonical(k)
        source_val = source_record.props.get(k)
        if str(notion_val) != str(source_val):
            res[k] = (notion_val, source_val)
    return res


def generate_modifiable_fields_from_notion(
    properties_in_notion: dict[str, str],
) -> dict[str, NotionType]:
    res: dict[str, NotionType] = {}
    for k, v in properties_in_notion.items():
        if k in {"created_by", "created_at", "last_edited_by", "last_edited_at"}:
            continue
        if v in {
            "formula",
            "rollup",
            "unique_id",
        }:
            continue
        res[k] = notion_type_from_str(v)
    return res


# pylint: disable=too-many-locals
async def find_changes(
    plugin: PluginInstance, notion_processor: NotionProcessor
) -> AsyncGenerator[Union[PageToRemove, PageToAdd, PageToUpdate], None]:
    await notion_processor.read_db_props()
    title_in_notion = notion_processor.db_info.title
    assert title_in_notion
    fields_intersection = generate_modifiable_fields_from_notion(
        notion_processor.db_info.properties
    )
    t0 = time.perf_counter()
    indexed_source_records, rows_count_in_source = parse_source(
        plugin=plugin,
        title_in_notion=title_in_notion,
        fields_intersection=fields_intersection,
    )
    stats.source_records.set(rows_count_in_source, time.perf_counter() - t0)
    logger.info(
        "read %d records from source in %.1fs",
        stats.source_records.count,
        round(stats.source_records.seconds, 1),
    )
    t0 = time.perf_counter()
    num_notion_records = 0
    with MyProgressBar(
        length=stats.source_records.count, label="Fetching Notion Records"
    ) as progress:
        async for notion_rec in notion_processor.iterate_over_pages():
            num_notion_records += 1
            progress.update(1)
            title = notion_rec.get_canonical(title_in_notion)
            if title is None:
                title = ""
            assert notion_rec.id
            title = title.strip()
            candidates = indexed_source_records.get(title)
            if candidates:
                candidate = pop_best_candidate(
                    notion_rec=notion_rec, candidates=candidates
                )
                if len(candidates) == 0:
                    del indexed_source_records[title]
                assert candidate
                # Now, we compare
                res = find_diffs(
                    notion_record=notion_rec,
                    source_record=candidate,
                    fields_to_compare=fields_intersection,
                )
                if res:
                    yield PageToUpdate(notion_rec.id, res)
            else:
                yield PageToRemove(notion_rec.id, title=title)
    stats.notion_records.set(num_notion_records, time.perf_counter() - t0)
    logger.info(
        "Processed %d records from Notion in %.1fs",
        stats.notion_records.count,
        round(stats.notion_records.seconds, 1),
    )
    for pages_with_same_title in indexed_source_records.values():
        for new_page in pages_with_same_title:
            yield PageToAdd(
                source_identifier=new_page.source_identifier,
                props_to_add=new_page.props,
            )


STOP_FETCHING = object()


class MyProgressBar:
    no_progress_bar = False

    def __init__(self, length: int, label: str) -> None:
        self.progress: Any = (
            None
            if MyProgressBar.no_progress_bar or length == 0
            else click.progressbar(length=length, label=label, file=sys.stderr)
        )

    def __enter__(self) -> "MyProgressBar":
        if self.progress:
            self.progress.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        if self.progress:
            self.progress.__exit__(exc_type, exc_value, tb)

    def update(self, n_steps: int) -> None:
        if self.progress:
            self.progress.update(n_steps=n_steps)


async def consume_updates(
    progress: MyProgressBar, q: asyncio.Queue[Union[object, Awaitable[Any]]]
) -> None:
    to_await = await q.get()
    while to_await is not STOP_FETCHING:
        assert isinstance(to_await, Awaitable)
        await to_await
        q.task_done()
        progress.update(1)
        to_await = await q.get()
    q.task_done()


async def process_modification(
    notion_processor: NotionProcessor,
    page_modification: NotionPageModification,
    confirmations: list[NotionPageModification],
    queue: asyncio.Queue[Union[object, Coroutine[Any, Any, Any]]],
) -> None:
    if isinstance(page_modification, PageToAdd):
        my_policy = notion_processor.apply_policies.add
    elif isinstance(page_modification, PageToUpdate):
        my_policy = notion_processor.apply_policies.update
    elif isinstance(page_modification, PageToRemove):
        my_policy = notion_processor.apply_policies.remove
    else:
        raise ValueError(f"Unexpected type: {type(page_modification)}")
    assert my_policy
    if my_policy == ApplyPolicy.APPLY:
        await queue.put(page_modification.apply_changes(notion_processor))
    elif my_policy == ApplyPolicy.CONFIRM:
        confirmations.append(page_modification)
    elif my_policy == ApplyPolicy.IGNORE:
        stats.ignored_modifications.increment(0)
    else:
        raise ValueError("Unexpected ApplyPolicy" + my_policy)


def filter_by_type(
    modification_type: str, confirmations: Iterable[NotionPageModification]
) -> Sequence[NotionPageModification]:
    return list(
        filter(lambda v: v.modification_type == modification_type, confirmations)
    )


def confirm_change(
    page_modification: NotionPageModification,
    possible_choices: Iterable[str],
    policy_by_modification_type: dict[str, str],
    items_with_same_type: Iterable[NotionPageModification],
) -> str:
    response = ""
    while response not in ["yes", "no"]:
        response = input(f"[CONFIRM] Pick a choice {possible_choices}:").lower()
        if response == "yes!":
            response = "yes"
            policy_by_modification_type[page_modification.modification_type] = "yes"
        elif response == "no!":
            response = "no"
            policy_by_modification_type[page_modification.modification_type] = "no"
        elif response == "see":
            print("Similar changes are", page_modification.modification_type)
            for modif in items_with_same_type:
                print(" - ", modif)
    return response


async def process_confirmations(
    notion_processor: NotionProcessor,
    confirmations: list[NotionPageModification],
    progress: MyProgressBar,
    queue: asyncio.Queue[Union[object, Coroutine[Any, Any, Any]]],
) -> None:
    policy_by_modification_type: dict[str, str] = {}

    while confirmations:
        page_modification = confirmations.pop(0)
        pre_set_policy = policy_by_modification_type.get(
            page_modification.modification_type
        )
        response = "unknown"
        if pre_set_policy:
            response = pre_set_policy
        else:
            possible_choices = ["yes", "no"]
            print(
                "[CONFIRM] Please confirm the current modification",
                page_modification,
            )
            items_with_same_type = filter_by_type(
                modification_type=page_modification.modification_type,
                confirmations=confirmations,
            )
            if len(items_with_same_type) > 0:
                possible_choices += ["see", "yes!", "no!"]
                print(
                    "\tThere are",
                    len(items_with_same_type),
                    "similar modification(s)",
                )
                print("\t\tsee\tlist similar modifications")
                print("\t\tyes!\tyes to all similar modifications")
                print("\t\tno!\tNo to all similar modifications")
            response = confirm_change(
                page_modification=page_modification,
                possible_choices=possible_choices,
                policy_by_modification_type=policy_by_modification_type,
                items_with_same_type=items_with_same_type,
            )
        assert response in ["yes", "no"]
        if response == "yes":
            print("  ✔", page_modification)
            await queue.put(page_modification.apply_changes(notion_processor))
        else:
            print("  🚫", page_modification)
            progress.update(1)


async def start_processing(
    plugin: PluginInstance,
    notion_processor: NotionProcessor,
) -> int:
    queue: asyncio.Queue[Union[object, Coroutine[Any, Any, Any]]] = asyncio.Queue()

    confirmations: list[NotionPageModification] = []
    updates = 0
    async for page_modification in find_changes(
        plugin=plugin, notion_processor=notion_processor
    ):
        updates += 1
        await process_modification(
            notion_processor=notion_processor,
            page_modification=page_modification,
            confirmations=confirmations,
            queue=queue,
        )

    with MyProgressBar(length=updates, label=f"Applying {updates} changes") as progress:
        await process_confirmations(
            notion_processor=notion_processor,
            confirmations=confirmations,
            progress=progress,
            queue=queue,
        )
        await queue.put(STOP_FETCHING)
        results = await asyncio.gather(
            asyncio.create_task(consume_updates(progress=progress, q=queue)),
            return_exceptions=True,
        )
        exceptions = []
        for res in results:
            if isinstance(res, BaseException):
                exceptions.append(res)
        if exceptions:
            print("[ERR] Had Exceptions", exceptions)

    return updates


_LOGGER_LEVELS = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]


_STATS_IN_CONSOLE = "console"


def change_behaviour_values() -> Iterable[str]:
    for x in ApplyPolicy:
        yield x.name


def parse_change_behaviour(value: str) -> ApplyPolicy:
    for x in ApplyPolicy:
        if value.upper() == x.name.upper():
            return x
    raise KeyError(
        f"{value} is not valid, must be any of {list(change_behaviour_values())}"
    )


def default_log_level() -> str:
    return os.getenv("NOTION_LOG_LEVEL", "WARNING")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export some data into a notion database"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--log-level",
        help=f"Set the default log level (default from $NOTION_LOG_LEVEL={default_log_level()})",
        choices=_LOGGER_LEVELS,
        default=default_log_level(),
    )
    parser.add_argument(
        "--notion-log-level",
        help="Set the log level for data2notion (default=INFO)",
        choices=_LOGGER_LEVELS,
        default="INFO",
    )
    parser.add_argument(
        "--no-progress-bar",
        help="Disable the progress bar",
        action="store_true",
    )
    parser.add_argument(
        "--notion-rate-limit",
        help="Set the notion rate-limiter, by default {default_rate_limit} (3 requests/sec, 100 initial bucket size)",
        default=default_rate_limit(),
    )
    parser.add_argument(
        "--statistics",
        help="Display Statistics when program ends",
        choices=[_STATS_IN_CONSOLE, "disabled"],
        default=_STATS_IN_CONSOLE,
    )
    parser.add_argument(
        "--notion-token",
        help="Notion Token to use $NOTION_TOKEN by default",
        default=os.getenv("NOTION_TOKEN"),
    )

    subparsers_root = parser.add_subparsers(
        title="action to perform",
        description="action to perform",
        help="sub-command help",
    )

    list_plugins_parser = subparsers_root.add_parser("plugins")
    list_plugins_parser.set_defaults(feat=display_plugins)

    write_to_notion_parser = subparsers_root.add_parser(
        "write-to-notion", help="write to Notion Database"
    )

    write_to_notion_parser.add_argument(
        "notion_database_id",
        help="Notion Database ID to sync with",
    )

    write_to_notion_parser.add_argument(
        "--add-policy",
        action="store",
        default=ApplyPolicy.APPLY,
        help="What do on new rows when writting to Notion (default: APPLY)",
        type=parse_change_behaviour,
        choices=list(change_behaviour_values()),
    )
    write_to_notion_parser.add_argument(
        "--delete-policy",
        action="store",
        default=ApplyPolicy.APPLY,
        help="What do on deleted rows when writting to Notion (default: APPLY)",
        type=parse_change_behaviour,
        choices=list(change_behaviour_values()),
    )
    write_to_notion_parser.add_argument(
        "--update-policy",
        action="store",
        default=ApplyPolicy.APPLY,
        help="What do on updated rows when writting to Notion (default: APPLY)",
        type=parse_change_behaviour,
        choices=list(change_behaviour_values()),
    )

    subparsers = write_to_notion_parser.add_subparsers(
        title="Plugins to read data",
        description="Read data from source using one of source types",
        help="Select import source from the list",
        required=True,
    )
    for plugin in available_plugins():
        if not plugin.is_disabled(__plugin_api_version__):
            p_plugin = subparsers.add_parser(
                plugin.info.name,
                description=plugin.info.description,
                help=plugin.info.description,
            )
            p_plugin.set_defaults(feat=plugin)
            plugin.register_in_parser(p_plugin)

    ns = parser.parse_args()
    if ns.no_progress_bar:
        MyProgressBar.no_progress_bar = True

    logging.basicConfig(
        level=ns.log_level,
        format="%(asctime)s [%(levelname)5s][%(name)11s] %(message)s",
        force=True,
    )
    logger.setLevel(ns.notion_log_level)

    global _rate_limiter  # pylint: disable=global-statement
    _rate_limiter = create_rate_limiter(ns.notion_rate_limit)

    try:
        if not hasattr(ns, "feat"):
            parser.print_help()
            return 0
        if callable(ns.feat):
            ns.feat()
        else:
            assert isinstance(ns.feat, Plugin)

            async def run_all() -> None:
                t0 = time.perf_counter()
                plugin_instance = ns.feat.start_parsing(ns)
                notion_processor = NotionProcessor(
                    ns.notion_database_id, ns.notion_token
                )
                notion_processor.apply_policies.add = ns.add_policy
                notion_processor.apply_policies.remove = ns.delete_policy
                notion_processor.apply_policies.update = ns.update_policy
                updates = await start_processing(
                    plugin_instance, notion_processor=notion_processor
                )
                t1 = time.perf_counter()
                print(
                    f"[DONE ] synchronized {ns.notion_database_id}, {updates} changes in {round(t1 - t0)}s"
                )

            asyncio.run(run_all())
    finally:
        # Also display stats in case of failure
        if ns.statistics == _STATS_IN_CONSOLE:
            print(f"[STATS] Statistics\n{stats.display_non_empty_stats()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
