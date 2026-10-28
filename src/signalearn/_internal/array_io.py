from pathlib import Path
from math import prod
import re
from zipfile import BadZipFile, ZipFile

import numpy as np

from .metadata import ARRAY_DTYPES, DISPLAY_ATTRIBUTES, prepare_colours, prepare_markers, validate_metadata
from .prediction import prepare_prediction
from .tensor import ArrayStorage, CoordinateReference, FileArray, check_shape, coordinate_specs, dimensions


def validate_array_schema(schema):
    from .io import schema_categories, validate_scalar_declarations

    required = {"version", "layout", "sample_fields", "fields"}
    if not required <= set(schema) or set(schema) - required - {"visible_fields"}:
        raise ValueError("Array schemas require version, layout, sample_fields, fields, and optional visible_fields")
    validate_scalar_declarations(schema["sample_fields"], allow_empty=True)
    fields = schema["fields"]
    if not isinstance(fields, dict) or not fields:
        raise ValueError("Array fields must be a non-empty object")
    if set(fields) & (set(schema["sample_fields"]) | {"id"}):
        raise ValueError("Array field names must not collide with scalar fields or 'id'")
    allowed = {"source", "dtype", "dims", "shape", "coords", "nullable", "prediction", *DISPLAY_ATTRIBUTES}
    for name, declaration in fields.items():
        try:
            if not isinstance(name, str) or not name or not isinstance(declaration, dict):
                raise ValueError("Expected a non-empty field name and an object declaration")
            if set(declaration) - allowed or not {"source", "dtype", "dims"} <= set(declaration):
                raise ValueError("Array declarations require source, dtype, dims, and supported attributes only")
            dims, _ = dimensions(declaration["dims"], declaration.get("shape"))
            if not dims:
                raise ValueError("Array dimensions must not be empty")
            coordinate_specs(declaration.get("coords"), dims)
            dtype = declaration["dtype"]
            validate_metadata(dtype, declaration.get("unit"), declaration.get("label"),
                              declaration.get("nullable", False), declaration.get("unit_latex"),
                              declaration.get("label_latex"), declaration.get("colour"),
                              declaration.get("marker"), array=True)
            prepare_colours(schema_categories(declaration, "colours"), dtype)
            prepare_markers(schema_categories(declaration, "markers"), dtype)
            prepare_prediction(declaration.get("prediction"), dims, declaration.get("coords"),
                               dtype=dtype, shape=declaration.get("shape"),
                               nullable=declaration.get("nullable", False), encoded=True)
            if declaration.get("nullable", False) and not dtype.startswith("float"):
                raise ValueError("Only floating-point arrays support nullable=True")
            source = declaration["source"]
            if not isinstance(source, dict) or set(source) - {"format", "path_column", "index_column", "key"}:
                raise ValueError("Invalid array source attributes")
            if source.get("format") not in {"npy", "npz"}:
                raise ValueError("Array source format must be 'npy' or 'npz'")
            if (source["format"] == "npz") != ("key" in source):
                raise ValueError("npz sources require a key; npy sources do not accept a key")
            if "path_column" not in source or any(not isinstance(value, str) or not value for value in source.values()):
                raise ValueError("Source names and columns must be non-empty strings")
        except (ValueError, TypeError) as error:
            raise ValueError(f"Field {name!r}: {error}") from error
    if "id" in source_columns(schema) and "id" not in schema["sample_fields"]:
        raise ValueError("An id column must be declared as a scalar sample field")
    visited, visiting = set(), set()

    def visit(name):
        if name in visiting:
            raise ValueError("Coordinate references cannot form cycles")
        if name in visited:
            return
        visiting.add(name)
        declaration = fields[name]
        for specification in coordinate_specs(declaration.get("coords"), declaration["dims"]).values():
            if isinstance(specification, CoordinateReference):
                target = fields.get(specification.name)
                if target is None or len(target["dims"]) != 1 or target["dtype"] == "bool":
                    raise ValueError(f"Field {name!r}: coordinate must reference a one-dimensional numeric array field")
                visit(specification.name)
        visiting.remove(name)
        visited.add(name)

    for name in fields:
        visit(name)
    if "visible_fields" in schema:
        visible = schema["visible_fields"]
        available = set(fields) | set(schema["sample_fields"]) | {"id"}
        if (not isinstance(visible, list) or not all(isinstance(name, str) for name in visible)
                or len(set(visible)) != len(visible) or not set(visible) <= available
                or not (set(schema["sample_fields"]) | {"id"}) <= set(visible)):
            raise ValueError("visible_fields must list unique declared fields and include all scalar fields and id")
        reachable = set()

        def dependencies(name):
            if name in reachable:
                return
            reachable.add(name)
            for value in coordinate_specs(fields[name].get("coords"), fields[name]["dims"]).values():
                if isinstance(value, CoordinateReference):
                    dependencies(value.name)

        for name in visible:
            if name in fields:
                dependencies(name)
        if set(fields) - reachable:
            raise ValueError("Hidden fields must be coordinate dependencies of visible fields")


def source_columns(schema):
    if schema["layout"] == "rows":
        return set()
    return {source[key] for declaration in schema["fields"].values()
            for source in [declaration["source"]] for key in ("path_column", "index_column") if key in source}


def array_header(path, format, key):
    if path.suffix.lower() != f".{format}":
        raise ValueError(f"Expected a .{format} array file")
    if format == "npy":
        with path.open("rb") as stream:
            return read_array_header(stream, path.stat().st_size)
    with ZipFile(path) as archive:
        member = f"{key}.npy"
        if archive.namelist().count(member) != 1:
            raise ValueError(f"Archive must contain exactly one member for key {key!r}")
        with archive.open(member) as stream:
            return read_array_header(stream, archive.getinfo(member).file_size)


def read_array_header(stream, file_size):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version in {(2, 0), (3, 0)}:
        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"Unsupported NPY header version {version!r}")
    if dtype.name not in ARRAY_DTYPES or dtype.fields is not None:
        raise ValueError("Only plain real numeric and boolean array dtypes are supported")
    if file_size < stream.tell() + prod(shape) * dtype.itemsize:
        raise ValueError("Array payload is shorter than its declared shape")
    return tuple(shape), dtype


def array_storage(manifest, columns, declaration, ids, name, headers):
    source = declaration["source"]
    items = []
    for row, identity in enumerate(ids):
        context = f"{manifest}: row {row + 2}, sample {identity!r}, field {name!r}"
        try:
            path_value = columns[source["path_column"]][row]
            if not isinstance(path_value, str) or not path_value:
                raise ValueError("Array source paths must be non-empty strings")
            path = Path(path_value)
            path = (manifest.parent / path).resolve() if not path.is_absolute() else path
            context += f", source {str(path)!r}"
            header_key = (str(path), source["format"], source.get("key"))
            if header_key not in headers:
                headers[header_key] = array_header(path, source["format"], source.get("key"))
            shape, dtype = headers[header_key]
            if dtype.name not in ARRAY_DTYPES or dtype.fields is not None:
                raise ValueError("Only plain real numeric and boolean array dtypes are supported")
            if dtype.name != declaration["dtype"]:
                raise ValueError(f"Stored dtype {dtype.name!r} does not match declared {declaration['dtype']!r}")
            index = None
            if "index_column" in source:
                value = columns[source["index_column"]][row]
                if re.fullmatch(r"[0-9]+", str(value)) is None:
                    raise ValueError("Array indices must be non-negative decimal integers")
                index = int(value)
                if not shape or index >= shape[0]:
                    raise ValueError("Array source index is out of bounds")
                shape = shape[1:]
            check_shape(shape, declaration["dims"], declaration.get("shape"), context)
            items.append(FileArray(str(path), source["format"], source.get("key"), index, shape, dtype, context))
        except (OSError, ValueError, KeyError, EOFError, BadZipFile) as error:
            raise ValueError(f"{context}: {error}") from error
    return ArrayStorage(tuple(items), np.dtype(declaration["dtype"]))
