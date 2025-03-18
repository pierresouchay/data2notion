import argparse
import datetime as dt
import hashlib
import json
import os
from typing import Any, Iterable

import httpx

from data2notion.serialization import NotionType

from . import __plugins__version__
from .plugin import Plugin, PluginInfo, PluginInstance, SourceRecord

_INVALID_QUERY_PFX = 'invalid parameter "query": '


def extract_error_message_from_prometheus(query: str, response: httpx.Response) -> str:
    if response.text:
        try:
            prom_err_msg = json.loads(response.text)

            err_msg = prom_err_msg.get("error", "")
            if err_msg.startswith(_INVALID_QUERY_PFX):
                return f"Error in query '{query}': {err_msg[len(_INVALID_QUERY_PFX):]}"  # noqa: E203
            if err_msg:
                return err_msg
        except json.decoder.JSONDecodeError:
            pass
        return f"Unknow Prometheus error HTTP[{response.status_code}]: {response.text}"
    return f"Unknown Prometheus error HTTP[{response.status_code}]"


class PrometheusPluginInstance(PluginInstance):
    def __init__(  # pylint: disable=R0913,R0917
        self,
        prometheus_url: str,
        query: str,
        remove_column: Iterable[str],
        row_id_expression: str,
        column_name_mapping: Iterable[tuple[str, str]],
    ) -> None:
        self.query = query
        column_mappings: dict[str, str] = (
            {elem[0]: elem[1] for elem in column_name_mapping}
            if column_name_mapping
            else {}
        )
        self.removed_columns = set(remove_column) if remove_column else set()

        response = httpx.get(f"{prometheus_url}/api/v1/query", params={"query": query})
        if response.status_code != 200:
            raise ValueError(
                f"{prometheus_url} replied: {extract_error_message_from_prometheus(query=query, response=response)}"
            )
        assert response.status_code == 200
        results = response.json()["data"]["result"]

        # Build a list of all labelnames used.
        labelnames_set: set[str] = set()
        for result in results:
            labelnames_set.update(result["metric"].keys())

        # Canonicalize
        has_name_prop = "__name__" in labelnames_set
        if has_name_prop:
            labelnames_set.discard("__name__")
        self.evaluation: str = row_id_expression
        self.labelnames = list(sorted(labelnames_set))
        self.mappings = {
            col: column_mappings.get(col, col)
            for col in ["__notion_row_id__", "timestamp", "value"] + self.labelnames
            if col not in self.removed_columns
        }
        self.results = results

    def setup_from_notion(
        self, notion_title_name: str, fields_intersection: dict[str, NotionType]
    ) -> None:
        assert notion_title_name
        assert fields_intersection
        self.mappings["__notion_row_id__"] = notion_title_name

    def supports_property(
        self,
        prop_name: str,
        notion_type: NotionType,
        first_record: SourceRecord,
    ) -> bool:
        assert notion_type
        assert first_record
        return prop_name in self.mappings.values()

    def compute_default_name(self, result: dict[str, Any]) -> str:
        labels_str = (
            "{"
            + ",".join(
                [
                    f'{label}={result["metric"].get(label, "")}'
                    for label in self.labelnames
                    if label not in self.removed_columns
                ]
            )
            + "}"
        )
        if self.evaluation == "__default__":
            base_name = result["metric"].get("__name__")
            if not base_name:
                # We don't have basename
                h = hashlib.new("sha256")
                h.update(self.query.encode("utf8"))
                base_name = h.hexdigest()[0:16]

            return f"{base_name}{labels_str}"

        available_labels = dict(result["metric"])
        available_labels["labels_str"] = labels_str
        try:
            return str(
                eval(self.evaluation, None, available_labels)  # pylint: disable=eval-used
            )
        except NameError as err:
            msg = f"Error while evaluating --row-id-expression '{self.evaluation}' did fail: {err}."
            available_variables = ", ".join(sorted(available_labels.keys()))
            print("[ERROR][prometheus]: ", msg)
            print(
                "       [prometheus]: *** The following variables are available:",
                available_variables,
                "***",
            )
            print(
                "       [prometheus]: Please fix you expression in --row-id-expression or use __default__"
            )
            raise err

    def values(self) -> Iterable[SourceRecord]:
        for idx, result in enumerate(self.results):
            id_of_row = self.compute_default_name(result=result)
            row = {self.mappings["__notion_row_id__"]: id_of_row}
            val_id = self.mappings.get("value")
            if val_id:
                row[val_id] = result["value"][1]
            val_id = self.mappings.get("timestamp")
            if val_id:
                row[val_id] = dt.datetime.fromtimestamp(
                    result["value"][0], tz=dt.timezone.utc
                ).isoformat()
            for label in self.labelnames:
                val_id = self.mappings.get(label)
                if val_id:
                    row[val_id] = result["metric"].get(label, "")
            yield SourceRecord(row, source_id=f"[{idx}] {id_of_row}")

    def close(self) -> None:
        pass


class PrometheusPlugin(Plugin):
    def __init__(self) -> None:
        pass

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="prometheus",
            author="Pierre Souchay",
            description="Write Prometheus metrics to Notion",
            version=__plugins__version__,
        )

    def register_in_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "query",
            help="PromQL Prometheus query to perfom, aggregation supported",
        )
        parser.add_argument(
            "--prometheus-url",
            action="store",
            default=os.getenv("PROMETHEUS_URL", "http://localhost:9090"),
            help="The URL of Prometheus instance to query, default to $PROMETHEUS_URL or http://localhost:9090",
        )
        parser.add_argument(
            "--column-name-mapping",
            action="append",
            nargs=2,
            help=(
                "map a column into a specific name (timestamp, value + labels) into another name:"
                " --column-name-mapping value MyValueColumnInNotion (can be repeated). "
                "Title column is mapped automatically"
            ),
        )
        parser.add_argument(
            "--row-id-expression",
            default="__default__",
            help=(
                "First column value (default=__default__'): default will concatenate metric name and labels "
                "You can use python expression using labels, example: --row-id-expression 'f\"my_metrics{labels_str}\"'"
                "labels_str:= label concatened the prometheus way, but all label can be used for evalutation."
                "By default, it will use `__name__{labels_str}` if __name__ exists, otherwise, __name__ will be "
                "replace by a hash of the query (if you perform aggregation)."
            ),
        )
        parser.add_argument(
            "--remove-column",
            action="append",
            help="Remove a column, can be specified multiple times",
        )

    def start_parsing(self, ns: argparse.Namespace) -> PluginInstance:
        return PrometheusPluginInstance(
            prometheus_url=ns.prometheus_url,
            query=ns.query,
            remove_column=ns.remove_column,
            row_id_expression=ns.row_id_expression,
            column_name_mapping=ns.column_name_mapping,
        )
