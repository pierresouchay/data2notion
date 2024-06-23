import argparse
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional


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

    def get_property_type(self, prop_name: str) -> Optional[SourcePropertyType]:
        assert prop_name
        return None

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

    def register_in_parser(self, parser: argparse.ArgumentParser) -> None:
        """
        Register to parse arguments
        """
        raise NotImplementedError()

    def start_parsing(self, ns: argparse.Namespace) -> PluginInstance:
        assert ns
        raise NotImplementedError()
