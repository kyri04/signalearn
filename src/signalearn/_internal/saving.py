import json
from contextlib import contextmanager
from math import prod
from pathlib import Path

from dask import delayed
import numpy as np

from .io import validate_schema
from .metadata import DISPLAY_ATTRIBUTES
from .prediction import encoded_prediction
from .tensor import CoordinateReference, batch_task


def save_destination(path, *, kind="Dataset"):
    path = Path(path).expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{kind} destination already exists: {path}")
    return path


@contextmanager
def output_file(path):
    stream = path.open("xb")
    try:
        with stream:
            yield stream
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def scalar_text(value, dtype):
    if value is None:
        return ""
    if dtype == "bool":
        return "true" if value else "false"
    if dtype.startswith(("int", "uint")):
        return str(int(value))
    if dtype.startswith("float"):
        return repr(float(value))
    return value


def field_declaration(field):
    declaration = {"dtype": field.dtype, "nullable": field.nullable}
    if field.prediction is not None:
        declaration["prediction"] = encoded_prediction(field.prediction)
    for key in DISPLAY_ATTRIBUTES:
        value = getattr(field, key)
        if key in {"colours", "markers"}:
            value = {scalar_text(category, field.dtype): display for category, display in value.items()}
            if not value:
                continue
        if value is not None:
            declaration[key] = value
    if not field.dims:
        if field.dtype == "string" and (field._values == "").any().compute():
            declaration["encoding"] = "json"
        return declaration
    declaration["dims"] = list(field.dims)
    shape = field._shape
    if not len(field) and shape is None:
        shape = field._values.sample_shape
    if shape is not None:
        declaration["shape"] = list(shape)
    if field._coords:
        declaration["coords"] = {
            dim: {"field": value.name} if isinstance(value, CoordinateReference) else list(value)
            for dim, value in field._coords.items()
        }
    return declaration


def saved_schema(data, fields):
    scalars, arrays, used = {}, {}, set(fields)
    for name, field in fields.items():
        declaration = field_declaration(field)
        if not field.dims:
            scalars[name] = declaration
            continue
        index = len(arrays)
        while True:
            path_column, index_column = f"__array_{index}_path", f"__array_{index}_index"
            if not {path_column, index_column} & used:
                break
            index += 1
        used.update((path_column, index_column))
        declaration["source"] = {"format": "npy", "path_column": path_column, "index_column": index_column}
        arrays[name] = declaration
    schema = {"version": 1, "layout": "rows", "fields": scalars}
    if arrays:
        schema = {"version": 1, "layout": "arrays", "sample_fields": scalars,
                  "fields": arrays, "visible_fields": list(data._fields)}
    validate_schema(schema)
    return schema


def save_headers(schema):
    if schema["layout"] == "rows":
        return list(schema["fields"])
    return [*schema["sample_fields"],
            *(column for field in schema["fields"].values()
              for column in (field["source"]["path_column"], field["source"]["index_column"]))]


def save_batches(fields, length):
    arrays = [field for field in fields.values() if field.dims]
    start, size = 0, 0
    for index in range(length):
        estimate = sum(prod(field._values.items[index].shape) * np.dtype(field.dtype).itemsize for field in arrays)
        if index > start and (index - start >= 1024 or size + estimate > 8 * 1024**2):
            yield range(start, index)
            start, size = index, 0
        size += estimate
    if start < length:
        yield range(start, length)


def save_tasks(fields, indices):
    return [batch_task(field, indices) if field.dims else delayed(field._values[indices.start:indices.stop])
            for field in fields.values()]


def save_array_batch(directory, number, start, values):
    if all(value.shape == values[0].shape for value in values):
        path = f"arrays/field_{number:04d}_{start:08d}.npy"
        np.save(directory / path, np.stack(values), allow_pickle=False)
        return [(path, index) for index in range(len(values))]
    result = []
    for index, value in enumerate(values):
        path = f"arrays/field_{number:04d}_{start + index:08d}.npy"
        np.save(directory / path, value[np.newaxis], allow_pickle=False)
        result.append((path, 0))
    return result


def save_rows(directory, fields, schema, indices, values):
    rows = [{} for _ in indices]
    scalars = schema["fields"] if schema["layout"] == "rows" else schema["sample_fields"]
    for number, ((name, field), batch) in enumerate(zip(fields.items(), values)):
        if field.dims:
            source = schema["fields"][name]["source"]
            locations = save_array_batch(directory, number, indices.start, batch)
            for row, (path, index) in zip(rows, locations):
                row.update({source["path_column"]: path, source["index_column"]: index})
        else:
            encoded = scalars[name].get("encoding") == "json"
            for row, value in zip(rows, batch):
                row[name] = (json.dumps(value, ensure_ascii=False) if encoded and value is not None
                             else scalar_text(value, field.dtype))
    return rows
