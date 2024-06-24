import argparse
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from data2notion.serialization import NotionType, serialize_canonical_from_source


class SourceRecord:
    """
    A wrapper for exposing records to data2notion.

    If the plugin exposes make a translation between properties from the source
    and the target, the values provided by props should be already translated.
    """

    def __init__(self, props: dict[str, str], source_id: Any) -> None:
        self._props = props
        self._source_identifier = source_id

    @property
    def props(self) -> dict[str, Any]:
        """
        Accessor to the properties
        """
        return self._props

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
    """
    PluginInstance is the instance of a Plugin that actually does the job.

    It is instanciated by `Plugin.start_parsing()` when all parameters
    have already been set.

    The calls are performed in this order. Calls marked as [Optional] don't need implementation.

    1. Call `PluginInstance.setup_from_notion()` to give hints about the structure
       of the target Notion Database. [Optional]
    2. Calls `PluginInstance.supports_property()` for optimizations to know
       if some fields from source could be discarded. [Optional]
    3. Call `PluginInstance.values()` to iterate on values provided by the source. [Mandatory]
    4. Several calls on `PluginInstance.get_property_as_canonical_value()` to perform diffs
       between Notion and the source. Default implementation might be fine, but plugins are
       free to re-implement those if types from source might be complicated to translate. [Optional]
    5. `PluginInstance.close()` is called when operations are all performed to release resources. [Optional].
    """

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
        """
        Get a property and translates it into a Canonical value: a value as a string.

        arrays are changed into multi-lines string separated by \n chars.

        A plugin implementation might choose to re-implement this if several possible
        values might be transtyped. Unsually, you don't need to re-implement that.
        """
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
        """
        Generator to return the records from the source
        """
        raise NotImplementedError()

    def close(self) -> None:
        """
        You can free the resources of the Plugin here if needed.

        This method will be systematically called.

        Usually, you might do this at the end of values, but provided in case of.
        """
        pass


@dataclass
class PluginInfo:
    """
    Informations about a plugin

    name: will be used in command line to use the plugin
    author: only informative
    description: help of plugin, as seen in help of command line
    version: version of plugin, informative only
    """

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
        Returns a String is plugin could not be enable with explaination.

        Called just after constructor.

        A plugin might be disabled, for instance if a lib requirement is not found.
        This enables us to provide many plugins without the need for distributing the
        various dependencies and have a lightweight distribution, for instance, a plugin
        implementing a database connection can choose to be disabled if the database driver
        is not present.

        Another reason might be a future deprecation to refuse working if a change in
        `plugin_api_version` has changed.
        """
        assert plugin_api_version
        return None

    @abstractmethod
    def register_in_parser(self, parser: argparse.ArgumentParser) -> None:
        """
        Register to parse arguments.

        This is how a plugin can hook in the command line.

        The command line already started a sub_parser, so, you just have
        to add all your parameters here.
        """
        raise NotImplementedError()

    @abstractmethod
    def start_parsing(self, ns: argparse.Namespace) -> PluginInstance:
        """
        Start parsing the source and instantiate the PluginInstance.

        This is where you might open connections to an API, read a file...
        """
        assert ns
        raise NotImplementedError()
