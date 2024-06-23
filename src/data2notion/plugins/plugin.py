import argparse
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional

from data2notion.serialization import NotionType, serialize_canonical_from_source


class SourcePropertyType(str, Enum):
    STR = "STR"


class SourceRecord:
    def __init__(self, props: dict[str, str], source_id: Any) -> None:
        self._props = props
        self._source_identifier = source_id

    @property
    def props(self) -> dict[str, Any]:
        return self._props

    def get(self, key: str) -> Any:
        return self.props.get(key)

    @property
    def source_identifier(self) -> Any:
        """
        Identificator in the source for debug purposes.

        In a CSV file, would be the line, the PK in a datadabase...
        """
        return self._source_identifier

    def __str__(self) -> str:
        return f"{self.props}"


class PluginInstance:
    def __init__(self) -> None:
        pass

    def setup_from_notion(
        self, notion_title_name: str, fields_intersection: dict[str, NotionType]
    ) -> None:
        """
        Called after initialization to find some type-hints about Notion Database.

        Might be use to tune the naming of props the plugin will return.
        """
        assert notion_title_name, fields_intersection
        pass

    def get_property_as_canonical_value(
        self, rec: SourceRecord, prop_name: str, notion_type: NotionType
    ) -> Optional[str]:
        vx = rec.props.get(prop_name)
        if vx is None:
            return None
        return serialize_canonical_from_source(vx, notion_type=notion_type)

    def supports_property(
        self,
        prop_name: str,
        notion_type: NotionType,
        first_record: SourceRecord,
    ) -> bool:
        """
        Optional To remplement for performance. The plugin can tell if it supports a given property.
        """
        assert prop_name
        assert notion_type
        assert first_record
        return True

    @abstractmethod
    def values(self) -> Iterable[SourceRecord]:
        raise NotImplementedError()

    def close(self) -> None:
        pass


@dataclass
class PluginInfo:
    name: str
    author: str
    description: str
    version: str


class Plugin:
    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        """
        Get the version of plugin and its version
        """
        raise NotImplementedError()

    def is_disabled(self, plugin_api_version: float) -> Optional[str]:
        """
        Returns a String is plugin could not be enable with explaination
        """
        assert plugin_api_version
        return None

    @abstractmethod
    def register_in_parser(self, parser: argparse.ArgumentParser) -> None:
        """
        Register to parse arguments
        """
        raise NotImplementedError()

    @abstractmethod
    def start_parsing(self, ns: argparse.Namespace) -> PluginInstance:
        assert ns
        raise NotImplementedError()
