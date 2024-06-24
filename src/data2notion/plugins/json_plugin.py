import argparse
import json
import sys
from typing import Iterable

from data2notion.serialization import NotionType

from . import __plugins__version__
from .plugin import Plugin, PluginInfo, PluginInstance, SourceRecord


class JSONPluginInstance(PluginInstance):
    def __init__(self, file: str, path_in_json: str) -> None:
        if file == "-":
            fd = sys.stdin
        else:
            fd = open(file, "rt", encoding="utf8")  # pylint: disable=R1732
        try:
            data = json.load(fd)
            for p in path_in_json.split("."):
                if p:
                    data = data[p]
        finally:
            if fd != sys.stdin:
                fd.close()
        self.data = data
        if not isinstance(data, list):
            raise ValueError(
                f"'--json-path={path_in_json}' does not point to array data: {str(json.dumps(data))[0:32]}…"
            )

        if len(data) > 0:
            self.props = data[0].keys()
        else:
            self.props = []

    def supports_property(
        self, prop_name: str, notion_type: NotionType, first_record: SourceRecord
    ) -> bool:
        return (
            self.get_property_as_canonical_value(
                rec=first_record, prop_name=prop_name, notion_type=notion_type
            )
            is not None
        )

    def values(self) -> Iterable[SourceRecord]:
        for idx, row in enumerate(self.data):
            yield SourceRecord(row, source_id=idx)

    def close(self) -> None:
        pass


class JSONPlugin(Plugin):
    def __init__(self) -> None:
        pass

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="json",
            author="Pierre Souchay",
            description="Write to Notion DB from a JSON file containing an array",
            version=__plugins__version__,
        )

    def register_in_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "json_file",
            help="JSON file to inject in Notion, if '-' is set, read from stdin",
        )
        parser.add_argument(
            "--json-path",
            dest="path_in_json",
            action="store",
            default=".",
            help="JSON path separated by dots to look for the array, example: calendar.appointments ",
        )

    def start_parsing(self, ns: argparse.Namespace) -> PluginInstance:
        return JSONPluginInstance(ns.json_file, path_in_json=ns.path_in_json)
