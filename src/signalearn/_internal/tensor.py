from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from zipfile import BadZipFile

from dask import delayed
import dask.array as da
import numpy as np

from .metadata import ARRAY_DTYPES
from .completion import install_ipython_completion
from .prediction import validate_prediction_batch, validate_prediction_values


@dataclass(frozen=True)
class CoordinateReference:
    name: str


@dataclass(frozen=True)
class FileArray:
    path: str
    format: str
    key: str | None
    index: int | None
    shape: tuple[int, ...]
    dtype: np.dtype
    context: str


@dataclass(frozen=True)
class IndexedArray:
    source: object
    axes: tuple
    shape: tuple[int, ...]
    dtype: np.dtype


@dataclass(frozen=True)
class ArrayStorage:
    items: tuple
    dtype: np.dtype
    sample_shape: tuple[int, ...] | None = None

    @property
    def shape(self):
        return (len(self.items),)

    def __getitem__(self, indices):
        return ArrayStorage(tuple(self.items[int(index)] for index in indices), self.dtype, self.sample_shape)


def dimensions(dims, shape):
    if isinstance(dims, (str, bytes)) or not isinstance(dims, Sequence):
        raise TypeError("dims must be a sequence of dimension names")
    dims = tuple(dims)
    if any(not isinstance(name, str) or not name for name in dims) or len(set(dims)) != len(dims):
        raise ValueError("Dimension names must be unique non-empty strings")
    if shape is not None:
        if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence) or len(shape) != len(dims):
            raise ValueError("shape must have one entry per dimension")
        if any(value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value < 0)
               for value in shape):
            raise ValueError("Shape entries must be non-negative integers or None")
        shape = tuple(None if value is None else int(value) for value in shape)
    if not dims and shape is not None:
        raise ValueError("Scalar fields do not take a shape constraint")
    return dims, shape


def check_shape(actual, dims, expected, context="Array field"):
    if len(actual) != len(dims):
        raise ValueError(f"{context}: rank {len(actual)} does not match dimensions {dims!r}")
    if expected is not None and any(size is not None and size != actual[index]
                                    for index, size in enumerate(expected)):
        raise ValueError(f"{context}: shape {actual!r} does not match {expected!r}")


def validate_array(value, dtype, nullable, shape, context="Array field"):
    if np.ma.isMaskedArray(value):
        raise ValueError(f"{context}: masked arrays require an explicit supported mask representation")
    value = np.asarray(value)
    if value.dtype.name != dtype:
        raise ValueError(f"{context}: dtype {value.dtype.name!r} does not match {dtype!r}")
    if value.shape != shape:
        raise ValueError(f"{context}: shape {value.shape!r} does not match {shape!r}")
    if value.dtype.kind == "f":
        if np.isinf(value).any() or (not nullable and np.isnan(value).any()):
            raise ValueError(f"{context}: infinite or undeclared missing array values")
    return np.array(value, copy=True)


def prepare_arrays(values, dtype, nullable, dims, shape):
    install_ipython_completion()
    if np.ma.isMaskedArray(values):
        raise TypeError("Masked arrays require an explicit supported mask representation")
    hint = None
    if isinstance(values, ArrayStorage):
        items = values.items
        inferred = values.dtype.name
        hint = values.sample_shape
    elif isinstance(values, (np.ndarray, da.Array)):
        if values.ndim != len(dims) + 1 or not all(np.isfinite(values.shape)):
            raise ValueError("Array fields require a known sample axis followed by their declared dimensions")
        inferred = values.dtype.name
        hint = tuple(values.shape[1:])
        if isinstance(values, np.ndarray):
            values = np.array(values, copy=True)
            values.flags.writeable = False
        items = tuple(values[index] for index in range(int(values.shape[0])))
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        if any(np.ma.isMaskedArray(value) for value in values):
            raise TypeError("Masked arrays require an explicit supported mask representation")
        items = tuple(value if isinstance(value, da.Array) else np.array(value, copy=True) for value in values)
        inferred = items[0].dtype.name if items else None
    else:
        raise TypeError("Array field values must contain one numeric array per sample")
    dtype = inferred if dtype is None else dtype
    if dtype not in ARRAY_DTYPES:
        raise ValueError("Supply a supported numeric array dtype, including for empty array fields")
    nullable = False if nullable is None else nullable
    if nullable and np.dtype(dtype).kind != "f":
        raise ValueError("Only floating-point array fields support nullable=True")
    for item in items:
        context = item.context if isinstance(item, FileArray) else "Array field"
        if item.dtype.name != dtype:
            raise ValueError(f"{context}: array dtype must match {dtype!r}; no casting is inferred")
        if not all(isinstance(size, (int, np.integer)) and size >= 0 for size in item.shape):
            raise ValueError(f"{context}: array shapes must be known")
        check_shape(item.shape, dims, shape, context)
        if isinstance(item, np.ndarray):
            validate_array(item, dtype, nullable, item.shape, context)
            item.flags.writeable = False
    if items:
        hint = tuple(items[0].shape) if all(item.shape == items[0].shape for item in items) else None
    return ArrayStorage(items, np.dtype(dtype), hint), dtype, nullable


def coordinate_specs(coords, dims):
    if coords is None:
        return MappingProxyType({})
    if not isinstance(coords, Mapping) or set(coords) - set(dims):
        raise ValueError("coords must map declared dimensions to coordinate values or field references")
    result = {}
    for name, value in coords.items():
        if isinstance(value, Mapping):
            if set(value) != {"field"} or not isinstance(value["field"], str) or not value["field"]:
                raise ValueError("Coordinate references require exactly one non-empty field name")
            result[name] = CoordinateReference(value["field"])
        else:
            if np.ma.isMaskedArray(value):
                raise ValueError("Coordinate masks require an explicit supported representation")
            array = np.asarray(value)
            if array.ndim != 1 or array.dtype.kind not in "biufU":
                raise ValueError("Inline coordinates must be numeric, boolean, or string vectors")
            if array.dtype.kind == "f" and not np.isfinite(array).all():
                raise ValueError("Coordinates must not be missing or infinite")
            if array.dtype.kind != "b" and any(isinstance(item, (bool, np.bool_)) for item in value):
                raise ValueError("Inline coordinates must have one scalar kind")
            if array.dtype.kind == "U" and any(not isinstance(item, str) for item in value):
                raise ValueError("Inline coordinates must have one scalar kind")
            result[name] = tuple(array.tolist())
    return MappingProxyType(result)


def check_coordinates(field):
    for dim, specification in field._coords.items():
        axis = field.dims.index(dim)
        if isinstance(specification, CoordinateReference):
            coordinate = field._coordinate_fields.get(dim)
            if coordinate is None:
                continue
            if len(coordinate.dims) != 1 or coordinate.dtype == "bool":
                raise ValueError(f"Coordinate {dim!r} must reference a one-dimensional numeric array field")
            if coordinate.ids != field.ids:
                raise ValueError("Coordinate fields must have matching sample IDs")
            lengths = [item.shape[0] for item in coordinate._values.items]
        else:
            lengths = [len(specification)] * len(field.ids)
        for item, length in zip(field._values.items, lengths):
            if item.shape[axis] != length:
                context = item.context if isinstance(item, FileArray) else f"Field {field.name!r}"
                raise ValueError(f"{context}, coordinate {dim!r}: length {length} does not match axis length {item.shape[axis]}")


def bind_coordinates(fields):
    from .data import clone_field

    resolved, visiting = {}, set()

    def resolve(name):
        if name in resolved:
            return resolved[name]
        if name in visiting:
            raise ValueError("Coordinate references cannot form cycles")
        visiting.add(name)
        field = fields[name]
        if field._coords:
            bindings = {}
            for dim, specification in field._coords.items():
                if isinstance(specification, CoordinateReference):
                    if specification.name in fields:
                        bindings[dim] = resolve(specification.name)
                    elif dim in field._coordinate_fields:
                        bindings[dim] = field._coordinate_fields[dim]
                    else:
                        raise ValueError(f"Field {name!r}: unknown coordinate field {specification.name!r}")
            field = clone_field(field)
            object.__setattr__(field, "_coordinate_fields", MappingProxyType(bindings))
            check_coordinates(field)
        visiting.remove(name)
        resolved[name] = field
        return field

    for name in fields:
        resolve(name)
    return {name: resolved[name] for name in fields}


def field_coordinates(field):
    from ..data import SampleField
    from .data import clone_field

    result = {}
    for dim, specification in field._coords.items():
        if isinstance(specification, CoordinateReference):
            if dim not in field._coordinate_fields:
                raise ValueError(f"Coordinate {dim!r} is unresolved; construct a Dataset containing its field")
            coordinate = clone_field(field._coordinate_fields[dim])
            object.__setattr__(coordinate, "_dataset", field._dataset)
            if coordinate.nullable:
                object.__setattr__(coordinate, "nullable", False)
            result[dim] = SampleField(coordinate) if isinstance(field, SampleField) else coordinate
        else:
            values = np.array(specification)
            values.flags.writeable = False
            result[dim] = values
    return MappingProxyType(result)


def read_file_array(path, format, key):
    try:
        if format == "npy":
            return np.load(path, mmap_mode="r", allow_pickle=False)
        with np.load(path, allow_pickle=False) as archive:
            return archive[key]
    except (OSError, ValueError, KeyError, EOFError, BadZipFile) as error:
        raise ValueError(f"{path}: cannot read array {key or ''!r}: {error}") from error


def select_file_array(value, index, dtype, nullable, shape, context, selections=()):
    try:
        value = value if index is None else value[index]
        if value.shape != shape or value.dtype.name != dtype:
            raise ValueError(f"Source shape or dtype changed; expected {shape!r} and {dtype!r}")
        value = select_axes(value, selections)
        return validate_array(value, dtype, nullable, value.shape, context)
    except (ValueError, IndexError, TypeError) as error:
        raise ValueError(f"{context}: {error}") from error


def check_coordinate_values(value, coordinates, dims, context):
    for name, coordinate in coordinates.items():
        coordinate = np.asarray(coordinate)
        if coordinate.ndim != 1 or coordinate.shape[0] != value.shape[dims.index(name)]:
            raise ValueError(f"{context}: coordinate {name!r} does not match array axis")
        if coordinate.dtype.kind == "f" and not np.isfinite(coordinate).all():
            raise ValueError(f"{context}: coordinate {name!r} must not be missing or infinite")
    return value


def array_base(item):
    selections = []
    while isinstance(item, IndexedArray):
        selections.append(item.axes)
        item = item.source
    return item, tuple(reversed(selections))


def select_axes(value, selections):
    for axes in selections:
        for axis, indices in axes:
            value = np.take(value, np.asarray(indices, dtype=np.intp), axis=axis)
    return value


def array_task(item, dtype, nullable, context):
    item, selections = array_base(item)
    if isinstance(item, FileArray):
        loaded = delayed(read_file_array, pure=True)(item.path, item.format, item.key)
        result = delayed(select_file_array)(loaded, item.index, dtype, nullable, item.shape, item.context, selections)
    else:
        item = select_axes(item, selections)
        result = delayed(validate_array)(item, dtype, nullable, tuple(item.shape), context)
    return result


def sample_task(field, index):
    item = field._values.items[index]
    context = item.context if isinstance(item, FileArray) else f"Field {field.name!r}, sample {field.ids[index]!r}"
    result = array_task(item, field.dtype, field.nullable, context)
    coordinates = {}
    for name, specification in field._coords.items():
        if isinstance(specification, CoordinateReference):
            if name not in field._coordinate_fields:
                raise ValueError(f"Coordinate {name!r} is unresolved; construct a Dataset containing its field")
            coordinates[name] = sample_task(field._coordinate_fields[name], index)
    if coordinates:
        result = delayed(check_coordinate_values)(result, coordinates, field.dims, context)
    if field.prediction is not None:
        result = delayed(validate_prediction_values)(result, field.prediction, field.dims)
    return result


def select_file_batch(value, items, dtype, nullable, selections):
    return tuple(select_file_array(value, item.index, dtype, nullable, item.shape, item.context, selection)
                 for item, selection in zip(items, selections))


def validate_memory_batch(values, dtype, nullable, shapes, contexts):
    return tuple(validate_array(value, dtype, nullable, shape, context)
                 for value, shape, context in zip(values, shapes, contexts))


def assemble_batch(groups, positions, length):
    result = [None] * length
    for values, locations in zip(groups, positions):
        for value, index in zip(values, locations):
            result[index] = value
    return tuple(result)


def check_batch_coordinates(values, coordinates, dims, contexts):
    for index, value in enumerate(values):
        check_coordinate_values(value, {name: arrays[index] for name, arrays in coordinates.items()},
                                dims, contexts[index])
    return values


def batch_task(field, indices):
    files, memory, memory_positions = {}, [], []
    for position, index in enumerate(indices):
        item, selections = array_base(field._values.items[index])
        if isinstance(item, FileArray):
            entries, positions, transforms = files.setdefault((item.path, item.format, item.key), ([], [], []))
            entries.append(item)
            positions.append(position)
            transforms.append(selections)
        else:
            memory.append(select_axes(item, selections))
            memory_positions.append(position)
    groups, positions = [], []
    for (path, format, key), (entries, locations, selections) in files.items():
        loaded = delayed(read_file_array, pure=True)(path, format, key)
        groups.append(delayed(select_file_batch)(loaded, delayed(tuple(entries), traverse=False),
                                                 field.dtype, field.nullable, delayed(selections, traverse=False)))
        positions.append(locations)
    if memory:
        groups.append(delayed(validate_memory_batch)(memory, field.dtype, field.nullable,
                      [tuple(item.shape) for item in memory],
                      [f"Field {field.name!r}, sample {field.ids[indices[position]]!r}" for position in memory_positions]))
        positions.append(memory_positions)
    result = delayed(assemble_batch)(groups, delayed(positions, traverse=False), len(indices))
    coordinates = {}
    for dim, specification in field._coords.items():
        if isinstance(specification, CoordinateReference):
            if dim not in field._coordinate_fields:
                raise ValueError(f"Coordinate {dim!r} is unresolved; construct a Dataset containing its field")
            coordinates[dim] = batch_task(field._coordinate_fields[dim], indices)
    if coordinates:
        contexts = [f"Field {field.name!r}, sample {field.ids[index]!r}" for index in indices]
        result = delayed(check_batch_coordinates)(result, coordinates, field.dims, contexts)
    if field.prediction is not None:
        result = delayed(validate_prediction_batch)(result, field.prediction, field.dims)
    return result


def materialise_arrays(field, indices=None):
    indices = tuple(range(len(field.ids)) if indices is None else indices)
    values = batch_task(field, indices).compute()
    if values and all(value.shape == values[0].shape for value in values):
        return np.stack(values)
    if not values:
        shape = field._values.sample_shape or field._shape
        if shape is not None and all(size is not None for size in shape):
            return np.empty((0, *shape), dtype=field.dtype)
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def tensor_preview(field, display):
    options = np.get_printoptions()
    shapes = [item.shape for item in field._values.items]
    size = sum(int(np.prod(shape)) for shape in shapes)
    if size <= options["threshold"] or not shapes:
        return display(materialise_arrays(field))
    edge = options["edgeitems"]
    indices = list(range(min(edge, len(shapes))))
    indices += list(range(max(len(indices), len(shapes) - edge), len(shapes)))
    values = materialise_arrays(field, indices) if indices else ()
    with np.printoptions(threshold=1, edgeitems=edge):
        content = ", ".join(display(value) for value in values)
    return f"array([{content}{', ' if content else ''}...], dtype={field.dtype}, samples={len(shapes)})"
