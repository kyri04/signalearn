from collections.abc import Sequence
from copy import deepcopy

from dask import delayed
import dask.array as da
import numpy as np

from ..data import Dataset, Field
from .data import align_field, require_scalar


def selected_fields(x):
    if isinstance(x, Dataset):
        fields = tuple(field for name, field in x.fields.items() if name != "id")
    elif isinstance(x, Field):
        fields = (x,)
    elif isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
        fields = tuple(x)
    else:
        raise TypeError("x must be a Field, a sequence of Fields, or a Dataset")
    if not fields or any(not isinstance(field, Field) for field in fields):
        raise TypeError("x must contain at least one SignaLearn Field")
    ids = fields[0].ids
    if not ids:
        raise ValueError("Model inputs must contain at least one sample")
    names = tuple(field.name or ("x" if len(fields) == 1 else f"x{index}")
                  for index, field in enumerate(fields))
    if len(set(names)) != len(names) or "id" in names:
        raise ValueError("Model input fields must have distinct names other than 'id'")
    return tuple(align_field(field, ids, allow_extra=True) for field in fields), names, ids


def native_column(values, dtype):
    if dtype == "float64":
        return np.asarray(values, dtype=float)
    if dtype == "int64":
        if any(value is None for value in values):
            return np.array([np.nan if value is None else int(value) for value in values], dtype=object)
        return np.asarray(values, dtype=np.int64)
    if dtype == "bool" and not any(value is None for value in values):
        return np.asarray(values, dtype=bool)
    return np.asarray(values, dtype=object)


def native_matrix(arrays, dtypes, vectors=None):
    vectors = vectors or (False,) * len(arrays)
    columns = [np.stack(values) if vector else native_column(values, dtype).reshape(-1, 1)
               for values, dtype, vector in zip(arrays, dtypes, vectors)]
    if len({column.dtype for column in columns}) == 1 and all(column.dtype.kind in "biuf" for column in columns):
        return np.column_stack(columns)
    matrix = np.empty((len(columns[0]), sum(column.shape[1] for column in columns)), dtype=object)
    start = 0
    for column in columns:
        matrix[:, start:start + column.shape[1]] = column
        start += column.shape[1]
    return matrix


def model_inputs(fields, target, parameters, ids, *, snapshot=False):
    from .features import feature_width
    from .tensor import batch_task

    for field in fields:
        feature_width(field)
    count = len(fields)
    fields = list(fields)
    target_index = None
    if target is not None:
        if not isinstance(target, Field):
            raise TypeError("target must be a SignaLearn Field")
        require_scalar(target, "Tabular model targets")
        target_index = len(fields)
        fields.append(align_field(target, ids, allow_extra=True))
    columns, options = {}, {}
    for name, value in parameters.items():
        if isinstance(value, Field):
            require_scalar(value, f"Native parameter {name!r}")
            columns[name] = len(fields)
            fields.append(align_field(value, ids, allow_extra=True))
        else:
            options[name] = deepcopy(value) if snapshot else value
    return delayed(native_inputs)([batch_task(field, tuple(range(len(field)))) if field.dims else field._values for field in fields],
                                  [field.dtype for field in fields], count,
                                  target_index, columns, options, [bool(field.dims) for field in fields[:count]])


def native_inputs(arrays, dtypes, count, target_index, columns, options, vectors=None):
    matrix = native_matrix(arrays[:count], dtypes[:count], vectors)
    target = None if target_index is None else native_target(arrays[target_index], dtypes[target_index])
    parameters = dict(options)
    for name, index in columns.items():
        parameters[name] = native_column(arrays[index], dtypes[index])
    return matrix, target, parameters


def native_target(values, dtype):
    if any(value is None for value in values):
        raise ValueError("Target values must not be missing")
    return native_column(values, dtype)


def output_column(values, index, dtype):
    column = values[:, index]
    if dtype != "float64":
        return np.array(column, copy=True)
    return np.array([None if np.isnan(value) else float(value) for value in column], dtype=object)


def output_storage(result, index, length, dtype):
    return da.from_delayed(delayed(output_column)(result, index, dtype), shape=(length,), dtype=object)


def output_dataset(result, schema, ids):
    output = {}
    for index, (name, attributes) in enumerate(schema.items()):
        if attributes.get("dims"):
            storage = da.from_delayed(result, shape=(len(ids), *attributes["shape"]), dtype=attributes["dtype"])
        else:
            storage = output_storage(result, index, len(ids), attributes["dtype"])
        output[name] = Field(storage, ids=ids, **attributes)
    return Dataset(output, ids=ids)
