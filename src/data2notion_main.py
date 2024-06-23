#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os
import random
import sys
import time
from abc import abstractmethod
from typing import Any, AsyncGenerator, Awaitable, Callable, Coroutine, Iterable, Union

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

__version__ = "0.0.1"

__plugin_api_version__ = 1.0


def available_plugins() -> Iterable[Plugin]:
    additional_plugins = os.getenv("DATA2NOTION_ADDITIONAL_PLUGINS", "").split(",")
    for plugin_spec in additional_plugins + [
        "plugins.csv_plugin:CSVPlugin",
        "plugins.json_plugin:JSONPlugin",
        "plugins.prometheus_plugin:PrometheusPlugin",
    ]:
        plugin_spec = plugin_spec.strip()
        if not plugin_spec:
            continue
        pkg, clz_name = plugin_spec.split(":")
        clz = __import__(
            pkg, globals={"__name__": "data2notion.main"}, fromlist=[clz_name], level=1
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


def retry_http() -> (
    Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]
):
    def wrapper(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapped(*args: Any, **kwargs: Any) -> T:
            for retry in range(1, 5):
                sleep_for = retry**2 + random.random() * retry
                try:
                    response = await func(*args, **kwargs)
                    return response
                except APIResponseError as err:
                    if err.code in [
                        APIErrorCode.RateLimited,
                        APIErrorCode.ConflictError,
                        APIErrorCode.ServiceUnavailable,
                    ]:
                        if err.code == APIErrorCode.RateLimited:
                            sleep_for = min(60, sleep_for)
                        logger.warning(
                            "[RETRY %d] will retry in %ds due to %s: %s",
                            retry,
                            sleep_for,
                            err.code,
                            str(err),
                        )
                        await asyncio.sleep(sleep_for)
                    else:
                        logger.error(
                            "[ERROR] Notion API Error: %s: %s",
                            err.code,
                            str(err),
                        )
                        raise err
            return response

        return wrapped

    return wrapper


class NotionRecord:
    def __init__(self, props: dict[str, Any]):
        self.id = props["id"]
        assert props["archived"] is False
        self._props = props["properties"]

    def get_canonical(self, prop_name: str) -> Any:
        return get_canonical_value_from_notion(self._props.get(prop_name))


class NotionDataBaseInfo:
    def __init__(self, retrieve_db_info: dict[str, Any]):
        self.properties: dict[str, str] = dict()
        self.title = ""
        for k, v in retrieve_db_info.get("properties", {}).items():
            if k:
                typ = v["type"]
                self.properties[k] = typ
                if typ == "title":
                    self.title = k


class NotionProcessor:
    def __init__(self, database_id: str, notion_token: str):
        self.notion = AsyncClient(auth=notion_token)
        self.database_id = database_id
        self.db_info = NotionDataBaseInfo({})

    async def read_db_props(self):
        db_info = await self.notion.databases.retrieve(database_id=self.database_id)
        self.db_info = NotionDataBaseInfo(db_info)

    async def iterate_over_pages(self) -> AsyncGenerator[NotionRecord, None]:
        async for rec in async_iterate_paginated_api(
            self.notion.databases.query, database_id=self.database_id
        ):
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
        self.props: dict[str, Any] = dict()
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
    indexed_source_records: dict[str, list[SourceRecordCanonical]] = dict()
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
                    "Will compare on fields %s",
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
    # TODO: compare with notion_rec
    return candidates.pop()


class NotionPageModification:
    def __init__(self) -> None:
        pass

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
        try:
            await notion_processor.notion.pages.update(
                page_id=self.page_id, archived=True
            )
        except APIResponseError as err:
            if err.code in [APIErrorCode.ValidationError]:
                # for some reason, it sometimes was already archived... ignore the error
                if "Can't edit block that is archived" in str(err):
                    return str(self)
            raise err
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
        payload = dict()
        for k, old_and_new_val in self.updates.items():
            prop_type_str = notion_processor.db_info.properties[k]
            prop_type = notion_type_from_str(prop_type_str)
            payload[k] = {
                prop_type.name: convert_value_to_notion(prop_type, old_and_new_val[1])
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
        payload = dict()
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
    res: dict[str, tuple[Any, Any]] = dict()
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
    res: dict[str, NotionType] = dict()
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
    t1 = time.perf_counter()
    logger.info(
        "read %d records from source in %.1fs", rows_count_in_source, round(t1 - t0, 1)
    )
    async for notion_rec in notion_processor.iterate_over_pages():
        title = notion_rec.get_canonical(title_in_notion)
        if title is None:
            title = ""
        assert notion_rec.id
        title = title.strip()
        candidates = indexed_source_records.get(title)
        if candidates:
            candidate = pop_best_candidate(notion_rec=notion_rec, candidates=candidates)
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
            to_remove = PageToRemove(notion_rec.id, title=title)
            yield to_remove
    for pages_with_same_title in indexed_source_records.values():
        for new_page in pages_with_same_title:
            yield PageToAdd(
                source_identifier=new_page.source_identifier,
                props_to_add=new_page.props,
            )


STOP_FETCHING = object()


async def consume_updates(q: asyncio.Queue[Union[object, Awaitable[Any]]]) -> None:
    to_await = await q.get()
    while to_await is not STOP_FETCHING:
        assert isinstance(to_await, Awaitable)
        await to_await
        q.task_done()
        to_await = await q.get()
    q.task_done()


async def start_processing(
    plugin: PluginInstance,
    notion_processor: NotionProcessor,
) -> None:
    num_concurrent = int(os.getenv("NOTION_MAX_CONCURRENT_CHANGES", "10"))
    queue: asyncio.Queue[Union[object, Coroutine[Any, Any, Any]]] = asyncio.Queue(
        maxsize=num_concurrent
    )
    consumers = [
        asyncio.create_task(consume_updates(queue)) for _ in range(num_concurrent)
    ]
    t0 = time.perf_counter()
    updates = 0
    async for f in find_changes(plugin=plugin, notion_processor=notion_processor):
        updates += 1
        if isinstance(f, NotionPageModification):
            await queue.put(f.apply_changes(notion_processor))
        else:
            print("Don't know what to do with {f}")
    for _ in range(num_concurrent):
        await queue.put(STOP_FETCHING)
    await asyncio.gather(*consumers)
    t1 = time.perf_counter()
    print(f"[DONE] synchronized, {updates} changes in {round(t1 - t0)}s")


def main() -> int:
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(
        description="Export some data into a notion database"
    )
    parser.add_argument(
        "--log-level",
        help="Set the log level (default=WARN)",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        default="WARNING",
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
    logging.getLogger().setLevel(ns.log_level)
    if not hasattr(ns, "feat"):
        parser.print_help()
        return 0
    if callable(ns.feat):
        ns.feat()
        return 0
    else:
        assert isinstance(ns.feat, Plugin)
        plugin_instance = ns.feat.start_parsing(ns)
        notion_processor = NotionProcessor(ns.notion_database_id, ns.notion_token)
        asyncio.run(
            start_processing(plugin_instance, notion_processor=notion_processor)
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
