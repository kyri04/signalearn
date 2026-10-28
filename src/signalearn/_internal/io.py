import csv
import json
import math
import re
from pathlib import Path

from .metadata import DTYPES, DISPLAY_ATTRIBUTES, prepare_colours, prepare_markers, validate_metadata
from .prediction import prepare_prediction

_INTEGER = re.compile(r"[+-]?[0-9]+\Z")
_FLOAT = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"Non-finite JSON constant {value!r} is not supported")


def read_schema(path: Path) -> dict:
    try:
        schema = json.loads(path.read_text(encoding="utf-8"),
                            object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"{path}: invalid schema: {error}") from error
    try:
        validate_schema(schema)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    return schema


def validate_schema(schema):
    if not isinstance(schema, dict):
        raise ValueError("Schema must be a JSON object")
    if type(schema.get("version")) is not int or schema["version"] != 1:
        raise ValueError("Expected schema version 1")
    layout = schema.get("layout")
    if layout == "arrays":
        from .array_io import validate_array_schema

        validate_array_schema(schema)
        return
    if layout == "recordings":
        raise NotImplementedError("The recordings import layout is not implemented yet")
    if layout != "rows":
        raise ValueError("Expected layout 'rows' or 'arrays'")
    if set(schema) != {"version", "layout", "fields"}:
        raise ValueError("Schema requires exactly version, layout, and fields")
    validate_scalar_declarations(schema["fields"])


def validate_scalar_declarations(fields, *, allow_empty=False):
    if not isinstance(fields, dict) or (not fields and not allow_empty):
        raise ValueError("fields must be an object of field declarations and must not be empty for rows")
    for name, declaration in fields.items():
        prefix = f"Field {name!r}"
        if not name or not isinstance(declaration, dict):
            raise ValueError(f"{prefix}: expected a non-empty name and an object declaration")
        if set(declaration) - {"dtype", "nullable", "missing_values", "encoding", "prediction", *DISPLAY_ATTRIBUTES}:
            raise ValueError(f"{prefix}: unknown schema attributes")
        dtype = declaration.get("dtype")
        if not isinstance(dtype, str) or dtype not in DTYPES:
            raise ValueError(f"{prefix}: expected dtype from {sorted(DTYPES)}")
        if "encoding" in declaration and (dtype != "string" or declaration["encoding"] != "json"):
            raise ValueError(f"{prefix}: encoding must be 'json' and is supported only for strings")
        try:
            validate_metadata(dtype, declaration.get("unit"), declaration.get("label"),
                              declaration.get("nullable", False), declaration.get("unit_latex"),
                              declaration.get("label_latex"), declaration.get("colour"),
                              declaration.get("marker"))
            prepare_colours(schema_categories(declaration, "colours"), dtype)
            prepare_markers(schema_categories(declaration, "markers"), dtype)
            prepare_prediction(declaration.get("prediction"), dtype=dtype,
                               nullable=declaration.get("nullable", False), encoded=True)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{prefix}: {error}") from error
        if not isinstance(declaration.get("nullable", False), bool):
            raise ValueError(f"{prefix}: nullable must be a boolean")
        tokens = declaration.get("missing_values", [])
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
            raise ValueError(f"{prefix}: missing_values must be a list of literal strings")
        if name == "id" and (dtype != "string" or declaration.get("nullable", False)):
            raise ValueError("Field 'id' must be a non-nullable string")


def parse_scalar(value: str, declaration: dict):
    if value == "" or value in declaration.get("missing_values", []):
        if not declaration.get("nullable", False):
            raise ValueError("missing value in a non-nullable field")
        return None
    dtype = declaration["dtype"]
    if dtype == "string":
        if declaration.get("encoding") == "json":
            try:
                decoded = json.loads(value)
            except ValueError as error:
                raise ValueError("expected a JSON-encoded string") from error
            if not isinstance(decoded, str):
                raise ValueError("expected a JSON-encoded string")
            return decoded
        return value
    if dtype == "bool":
        if value not in {"true", "false"}:
            raise ValueError("expected literal true or false")
        return value == "true"
    if dtype == "int64":
        if not _INTEGER.fullmatch(value):
            raise ValueError("expected an int64 number without separators or units")
        number = int(value)
        if not -(2**63) <= number < 2**63:
            raise ValueError("integer is outside the int64 range")
        return number
    if not _FLOAT.fullmatch(value):
        raise ValueError("expected a finite float64 number without separators or units")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("expected a finite float64 number")
    return number


def schema_categories(declaration, attribute):
    mapping = declaration.get(attribute)
    if mapping is None:
        return None
    if not isinstance(mapping, dict):
        raise ValueError(f"{attribute} must be a JSON object mapping categories to display strings")
    result = {}
    for key, value in mapping.items():
        dtype = declaration["dtype"]
        if dtype == "string":
            category = key
        elif dtype.startswith(("uint", "int")) and dtype != "int64":
            import numpy as np

            if not _INTEGER.fullmatch(key):
                raise ValueError(f"Expected an integer category for {dtype}")
            category = int(key)
            limits = np.iinfo(dtype)
            if not limits.min <= category <= limits.max:
                raise ValueError(f"Category is outside the {dtype} range")
        else:
            category = parse_scalar(key, {"dtype": dtype})
        if category in result:
            raise ValueError(f"Repeated {attribute} category after dtype conversion: {key!r}")
        result[category] = value
    return result


def field_options(declaration):
    result = {name: declaration.get(name) for name in DISPLAY_ATTRIBUTES}
    for attribute in ("colours", "markers"):
        result[attribute] = schema_categories(declaration, attribute)
    result["dtype"] = declaration["dtype"]
    result["nullable"] = declaration.get("nullable", False)
    result["prediction"] = prepare_prediction(declaration.get("prediction"), declaration.get("dims", ()),
                                              declaration.get("coords"), dtype=declaration["dtype"],
                                              shape=declaration.get("shape"),
                                              nullable=declaration.get("nullable", False), encoded=True)
    return result


def read_csv_columns(path, fields, expected):
    if path.suffix.lower() != ".csv":
        raise ValueError(f"{path}: only .csv tables and manifests are supported")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            headers = next(reader, None)
            if not headers or any(not name for name in headers) or len(set(headers)) != len(headers):
                raise ValueError(f"{path}: expected unique, non-empty CSV headers")
            if set(headers) != expected:
                raise ValueError(
                    f"{path}: CSV columns do not match schema; "
                    f"undeclared={sorted(set(headers) - expected)}, "
                    f"missing={sorted(expected - set(headers))}"
                )
            columns = {name: [] for name in headers}
            id_position = headers.index("id") if "id" in headers else None
            seen_ids = set()
            for row in reader:
                context = f"{path}: row {reader.line_num}"
                if len(row) != len(headers):
                    raise ValueError(f"{context}: expected {len(headers)} cells, got {len(row)}")
                if id_position is not None:
                    context += f", sample {row[id_position]!r}"
                for name, value in zip(headers, row):
                    try:
                        parsed = parse_scalar(value, fields[name]) if name in fields else value
                    except ValueError as error:
                        raise ValueError(f"{context}, field {name!r}: {error}") from error
                    if name == "id":
                        if parsed in seen_ids:
                            raise ValueError(f"{context}, field 'id': duplicate sample ID {parsed!r}")
                        seen_ids.add(parsed)
                    columns[name].append(parsed)
        except csv.Error as error:
            raise ValueError(f"{path}: row {reader.line_num}: malformed CSV: {error}") from error
        except UnicodeError as error:
            raise ValueError(f"{path}: expected UTF-8 CSV: {error}") from error
    return columns
