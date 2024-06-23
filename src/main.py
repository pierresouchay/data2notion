#!/usr/bin/env python3
import argparse
import asyncio
import os
from typing import Any, AsyncGenerator, Iterable, Union

from notion_client import AsyncClient
from notion_client.helpers import async_iterate_paginated_api

from data2notion.plugins.plugin import Plugin, PluginInstance, SourceRecord
from data2notion.serialization import (
    NotionType,
    read_canonical_from_notion,
    str_to_notion_type,
)

__version__ = "0.0.1"

__plugin_api_version__ = 1.0


def available_plugins() -> Iterable[Plugin]:
    for plugin_spec in ["plugins.csv_plugin:CSVPlugin"]:
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
            msg = "[OK]"
        print(
            f" {msg} {p.info.name}\t{p.info.version}\tby {p.info.author}\tfeat: {p.info.description}"
        )


def find_title(rec: SourceRecord, title_in_notion: str) -> str:
    if rec.props.get(title_in_notion):
        return title_in_notion
    raise ValueError(
        f"Don't know how to find Notion title: {title_in_notion} in {list(rec.props.keys())}"
    )


class NotionRecord:
    def __init__(self, props: dict[str, Any]):
        self.id = props["id"]
        self.props = props["properties"]

    def get(self, prop_name: str) -> Any:
        return self.props[prop_name]


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


def parse_source(
    plugin: PluginInstance, title_in_notion: str
) -> dict[str, list[SourceRecord]]:
    indexed_source_records: dict[str, list[SourceRecord]] = dict()
    title_in_source = ""
    try:
        for rec in plugin.values():
            if not title_in_source:
                title_in_source = find_title(
                    rec=rec, title_in_notion=title_in_notion
                ).strip()
            idx = rec.get(title_in_source)
            if idx is None:
                print(
                    f"WARN: could not find {title_in_source} in {rec.source_identifier}"
                )
            else:
                lst = indexed_source_records.get(idx)
                if lst is None:
                    lst = []
                    indexed_source_records[idx] = lst
                lst.append(rec)
        return indexed_source_records
    finally:
        plugin.close()


def find_best_candidate(
    notion_rec: NotionRecord, candidates: list[SourceRecord]
) -> SourceRecord:
    if len(candidates) == 1:
        return candidates.pop()
    # TODO
    raise ValueError("Don't know yet how to disambiguate")


class PageToRemove:
    def __init__(self, page_id):
        self.page_id = page_id


async def find_changes(
    plugin: PluginInstance, notion_processor: NotionProcessor
) -> AsyncGenerator[Union[PageToRemove], None]:
    await notion_processor.read_db_props()
    title_in_notion = notion_processor.db_info.title
    assert title_in_notion
    indexed_source_records = parse_source(
        plugin=plugin, title_in_notion=title_in_notion
    )
    fields_intersection = {
        k: str_to_notion_type(v) for k, v in notion_processor.db_info.properties.items()
    }
    print(fields_intersection)
    async for notion_rec in notion_processor.iterate_over_pages():
        title = read_canonical_from_notion(
            notion_rec.get(title_in_notion), NotionType.title_id
        )
        if title is None:
            title = ""
        title = title.strip()
        candidates = indexed_source_records.get(title)
        if candidates:
            candidate = find_best_candidate(
                notion_rec=notion_rec, candidates=candidates
            )
            if len(candidates) == 0:
                del indexed_source_records[title]
            assert candidate
            # Now, we compare

        else:
            yield PageToRemove(notion_rec.id)


async def start_processing(
    plugin: PluginInstance, notion_processor: NotionProcessor
) -> None:
    async for f in find_changes(plugin=plugin, notion_processor=notion_processor):
        print(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export some data into a notion database"
    )
    parser.add_argument(
        "--notion-token",
        help="Notion Token to use $NOTION_TOKEN by default",
        default=os.getenv("NOTION_TOKEN"),
    )
    parser.add_argument(
        "notion_database_id",
        help="Notion Database ID to sync with",
    )
    subparsers = parser.add_subparsers(
        title="Plugins to read data",
        description="valid subcommands",
        help="sub-command help",
    )
    p_plugins = subparsers.add_parser("plugins")
    p_plugins.set_defaults(feat=display_plugins)

    for plugin in available_plugins():
        if not plugin.is_disabled(__plugin_api_version__):
            p_plugin = subparsers.add_parser(plugin.info.name)
            p_plugin.set_defaults(feat=plugin)
            plugin.register_in_parser(p_plugin)

    ns = parser.parse_args()
    if callable(ns.feat):
        ns.feat()
    else:
        assert isinstance(ns.feat, Plugin)
        plugin_instance = ns.feat.start_parsing(ns)
        notion_processor = NotionProcessor(ns.notion_database_id, ns.notion_token)
        asyncio.run(
            start_processing(plugin_instance, notion_processor=notion_processor)
        )


if __name__ == "__main__":
    main()
