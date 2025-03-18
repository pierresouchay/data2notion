import datetime as dt
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, Union


# pylint: disable=invalid-name
class NotionCanonicalRepr(str, Enum):
    array = list
    boolean = bool
    datetime = dt.datetime
    number_r = float
    person = "person"
    relation_r = "relation"
    string = str


class NotionType(Enum):
    checkbox = ("checkbox", NotionCanonicalRepr.boolean)
    created_by = ("created_by", NotionCanonicalRepr.person)
    created_time = ("created_time", NotionCanonicalRepr.datetime)
    date = ("date", NotionCanonicalRepr.datetime)
    email = ("email", NotionCanonicalRepr.string)
    files = ("files", NotionCanonicalRepr.array)
    formula = ("formula", NotionCanonicalRepr.string)
    last_edited_by = ("last_edited_by", NotionCanonicalRepr.person)
    last_edited_time = ("last_edited_time", NotionCanonicalRepr.datetime)
    multi_select = ("multi_select", NotionCanonicalRepr.array)
    number = ("number", NotionCanonicalRepr.number_r)
    people = ("people", NotionCanonicalRepr.person)
    phone_number = ("phone_number", NotionCanonicalRepr.string)
    relation = ("relation", NotionCanonicalRepr.relation_r)
    rich_text = ("rich_text", NotionCanonicalRepr.string)
    rollup = ("rollup", NotionCanonicalRepr.string)
    select = ("select", NotionCanonicalRepr.string)
    status = ("status", NotionCanonicalRepr.string)
    title = ("title", NotionCanonicalRepr.string)
    unique_id = ("unique_id", NotionCanonicalRepr.number_r)
    url = ("url", NotionCanonicalRepr.string)

    def __repr__(self) -> str:
        return self.value[0]


def notion_type_from_str(val: str) -> NotionType:
    nt = getattr(NotionType, val)
    assert isinstance(nt, NotionType)
    return nt


@dataclass
class NotionSerialize:
    read_from_notion: Callable[[Any], Any]
    write_to_notion: Callable[[Any], Any]


def no_op(val: Any) -> Any:
    return val


T = TypeVar("T")


def truncate_large_value(val: T) -> Union[T, str]:
    """
    Notion has a limit of 2000 chars in the API, so, we cannot send more than 2k chars
    This is an issue for both comparing the changes and send the updated value.
    """
    if val and isinstance(val, str) and len(val) > 2000:
        return val[0:1999] + "…"
    return val


def write_to_notion_rich_text(value: Any) -> list[dict[str, Any]]:
    return [{"type": "text", "text": {"content": truncate_large_value(value)}}]


def read_from_rich_text(prop: Any) -> Any:
    return "\n".join(t["plain_text"] for t in prop) or None


def read_by_name(prop: Any) -> Any:
    if not isinstance(prop, dict):
        raise AssertionError(
            f"prop is supposed to be a dict, but was {type(prop)}: {prop}"
        )
    return prop.get("name", None)


def read_names(prop: Any) -> Any:
    if not isinstance(prop, list):
        raise AssertionError(
            f"prop is supposed to be a list, but was {type(prop)}: {prop}"
        )
    return [read_by_name(p) for p in prop]


def read_names_as_joined_str(prop: Any) -> str:
    return "\n".join(read_names(prop))


def read_date(prop: Any) -> Any:
    if prop is None:
        return None
    if isinstance(prop, str):
        return prop
    if not isinstance(prop, dict):
        raise AssertionError(
            f"prop is supposed to be a dict, but was {type(prop)}: {prop}"
        )
    assert prop["time_zone"] is None
    return prop["start"]


def read_files(prop: Any) -> list[str]:
    if not isinstance(prop, list):
        raise AssertionError(
            f"prop is supposed to be a dict, but was {type(prop)}: {prop}"
        )
    files = []
    for f in prop:
        # Other type is file, but cannot be saved as URL is only available during 3600s only
        if f["type"] == "external":
            url = f["external"]["url"]
            files.append(url)
    return files


def parse_formula(formula: dict[str, Any]) -> Any:
    subtype = formula["type"]
    if subtype in ["string", "number", "boolean"]:
        return formula[subtype]
    if subtype == "date":
        return read_date(formula["date"])
    raise NotImplementedError(f"unsupported formula: {json.dumps(formula)}")


NotionSerializeNoOp = NotionSerialize(read_from_notion=no_op, write_to_notion=no_op)


def str_to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return bool(v)
    if v in ["f", "False", "false", "0", "no", "n"]:
        return False
    return bool(v)


def write_number_canonic(number: Any) -> Any:
    if number == "" or number is None:
        return None
    x = float(number)
    if x.is_integer():
        # Write 1, not 1.0 to avoid too many Changes
        return int(x)
    return x


# Not implemented formats:
# status, formula, relation, rollup, people, files, created_time, created_by, last_edited_time, last_edited_by
serialization_writers = {
    NotionType.checkbox: str_to_bool,
    NotionType.date: lambda a: {"start": a} if a else None,
    NotionType.email: no_op,
    NotionType.multi_select: lambda a: (
        [{"name": x} for x in a.split("\n") if x] if a else []
    ),
    NotionType.number: write_number_canonic,
    NotionType.phone_number: no_op,
    NotionType.rich_text: write_to_notion_rich_text,
    NotionType.select: lambda a: {"name": a.replace(",", " ")} if a else None,
    NotionType.title: write_to_notion_rich_text,
    NotionType.url: no_op,
}


def serialize_for_notion_write(val: Any, notion_type: NotionType) -> Optional[Any]:
    if val is None:
        return None
    serializer = serialization_writers.get(notion_type)
    vx = str(val)
    if serializer is None:
        return vx
    return serializer(vx)


def read_rollup(rollup: Any) -> Any:
    if not isinstance(rollup, dict):
        raise AssertionError(
            f"prop is supposed to be a dict, but was {type(rollup)}: {rollup}"
        )
    subtype = rollup["type"]
    if subtype == "array":
        the_array = rollup["array"]
        if not isinstance(the_array, list):
            raise AssertionError(
                f"prop is supposed to be a dict, but was {type(the_array)}: {the_array}"
            )
        return [get_canonical_value_from_notion(val) for val in the_array]
    if subtype in ["number"]:
        return rollup[subtype]
    if subtype == "date":
        return read_date(rollup[subtype])
    raise NotImplementedError(f"unsupported rollup: {json.dumps(rollup)}")


def read_by_id(prop: dict[str, Any]) -> Any:
    return prop["id"]


def read_people(prop: list[dict[str, Any]]) -> list[str]:
    return [cid["id"] for cid in prop]


def read_bool(prop: Optional[bool]) -> Optional[bool]:
    return prop


def read_unique_id(prop: Any) -> str:
    return f"{prop['prefix']}-{prop['number']}"


serialization_readers: dict[NotionType, Callable[[Any], Any]] = {
    NotionType.checkbox: read_bool,
    NotionType.created_by: read_by_id,
    NotionType.created_time: read_date,
    NotionType.date: read_date,
    NotionType.email: no_op,
    NotionType.files: read_files,
    NotionType.formula: parse_formula,
    NotionType.last_edited_by: read_by_id,
    NotionType.last_edited_time: read_date,
    NotionType.multi_select: read_names_as_joined_str,
    NotionType.number: no_op,
    NotionType.people: read_people,
    NotionType.phone_number: no_op,
    NotionType.relation: no_op,
    NotionType.rich_text: read_from_rich_text,
    NotionType.rollup: read_rollup,
    NotionType.select: read_by_name,
    NotionType.status: read_by_name,
    NotionType.title: read_from_rich_text,
    NotionType.unique_id: read_unique_id,
    NotionType.url: no_op,
}


def convert_value_to_notion(prop_type: NotionType, val: Any) -> Any:
    try:
        conv = serialization_writers[prop_type]
        return conv(val)
    except KeyError as kerr:
        raise ValueError(f"Unknown type: {prop_type}") from kerr


def get_canonical_value_from_notion(prop: dict[str, Any]) -> Any:
    prop_type = prop["type"]
    assert prop_type
    try:
        val = prop[prop_type]
        if val is None:
            return None
        serial_id = notion_type_from_str(prop_type)
        res = serialization_readers[serial_id](val)
        if isinstance(res, list):
            return tuple(res)
        return res
    except AssertionError:
        logging.error("failed to read %s in %s", prop_type, prop)
        raise


def basic_type_to_str(
    field: Optional[Union[str, list, tuple, dt.datetime, bool]],
) -> str:
    if field is None:
        return ""
    if isinstance(field, (list, tuple)):
        return "\n".join([str(f) if f else "" for f in field])
    if isinstance(field, bool):
        return "true" if field else "false"
    if isinstance(field, (dt.datetime, dt.date)):
        return field.isoformat()
    return str(field)


def serialize_canonical_from_source(
    field: Optional[Union[str, list, tuple, dt.datetime, bool]], notion_type: NotionType
) -> str:
    val = basic_type_to_str(field=field)
    if notion_type in [NotionType.select, NotionType.multi_select]:
        return "\n".join(map(lambda v: v[0:100].replace(",", "_"), val.split("\n")))
    return truncate_large_value(str(val))


def read_canonical_from_notion(
    notion_prop: dict[str, Any], notion_type: NotionType
) -> Any:
    serializer = serialization_readers.get(notion_type)
    if serializer is None:
        serializer = no_op
    return serializer(notion_prop[notion_type.name])
