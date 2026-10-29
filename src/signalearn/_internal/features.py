from dask.base import tokenize
import numpy as np

from .data import infer_dtype
from .tensor import batch_task


def feature_width(field):
    if not field.dims:
        return 1
    if len(field.dims) != 1:
        raise TypeError("Sklearn features require scalar or one-dimensional vector Fields; reshape other arrays explicitly")
    shapes = {tuple(item.shape) for item in field._values.items}
    if len(shapes) != 1 or next(iter(shapes))[0] < 1:
        raise ValueError("Sklearn vector features require a fixed non-empty length across samples")
    if np.dtype(field.dtype).kind not in "biuf":
        raise TypeError("Sklearn vector features must be real numeric arrays")
    return int(next(iter(shapes))[0])


def feature_names(fields, names):
    result = tuple(column for field, name in zip(fields, names)
                   for column in ([f"{name}__{i}" for i in range(feature_width(field))] if field.dims else [name]))
    if len(set(result)) != len(result):
        raise ValueError("Expanded feature names collide; rename the selected Fields")
    return result


def feature_grid(field):
    from ..data import Field

    coordinate = field.coords.get(field.dims[0])
    if coordinate is None:
        return None
    if isinstance(coordinate, Field):
        unique = {}
        for index, item in enumerate(coordinate._values.items):
            unique.setdefault(tokenize(item), index)
        values = batch_task(coordinate, tuple(unique.values())).compute()
        if any(not np.array_equal(value, values[0]) for value in values[1:]):
            raise ValueError("Feature coordinates must match across samples; resample or prepare coordinates explicitly")
        array, unit = values[0], coordinate.unit
    else:
        array, unit = coordinate, None
    return {"values": np.asarray(array).tolist(), "dtype": infer_dtype(np.asarray(array, dtype=object)), "unit": unit}


def feature_contract(fields, names):
    return {name: {"dim": field.dims[0], "size": feature_width(field), "coords": feature_grid(field)}
            for field, name in zip(fields, names) if field.dims}


def preserving_transform(model):
    from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

    return type(model) in {StandardScaler, RobustScaler} or (type(model) is FunctionTransformer and model.func is None)


def vector_column(values, start, size, length):
    return np.asarray(values[:, start:start + size], dtype=np.float64).reshape(length, size)


def transform_dataset(result, schema, fields, names, ids):
    from types import MappingProxyType
    from dask import delayed
    import dask.array as da
    from ..data import Dataset, Field
    from .alignment import output_storage

    output = {}
    sources = dict(zip(names, fields))
    start = 0
    for name, attributes in schema.items():
        source = sources.get(name)
        size = attributes["shape"][0] if attributes.get("dims") else 1
        if attributes.get("dims"):
            storage = da.from_delayed(delayed(vector_column)(result, start, size, len(ids)),
                                      shape=(len(ids), size), dtype="float64")
        else:
            storage = output_storage(result, start, len(ids), attributes["dtype"])
        field = Field(storage, ids=ids, **attributes)
        if source is not None and field.dims:
            if any(coordinate.name in schema for coordinate in source._coordinate_fields.values()):
                raise ValueError("Transformed feature names collide with coordinate dependencies; select or rename inputs explicitly")
            object.__setattr__(field, "_coordinate_fields", MappingProxyType(dict(source._coordinate_fields)))
        output[name] = field
        start += size
    return Dataset(output, ids=ids)
