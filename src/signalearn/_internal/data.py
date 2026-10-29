from collections.abc import Mapping, Sequence
from keyword import iskeyword
import math
from numbers import Integral, Real
from types import MappingProxyType
from uuid import uuid4

import dask.array as da
import numpy as np

from .completion import install_ipython_completion
from .metadata import DISPLAY_ATTRIBUTES, updated_metadata

FIELD_ATTRIBUTES = ("_values", "_dims", "_shape", "_coords", "_coordinate_fields", "_dataset",
                    "ids", "dtype", "nullable", "name", "prediction", *DISPLAY_ATTRIBUTES)


def field_attributes(instance: object, fields: Mapping[str, object]) -> list[str]:
    names = set(object.__dir__(instance))
    names.update(name for name in fields if name.isidentifier() and not iskeyword(name))
    return sorted(names)


def materialise_values(field) -> np.ndarray:
    if field.dims:
        from .tensor import materialise_arrays

        return materialise_arrays(field)
    return np.array(field._values.compute(), copy=True)


def materialise_value(field, index: int) -> str | bool | int | float | None:
    if field.dims:
        from .tensor import sample_task

        return sample_task(field, index).compute()
    return np.asarray(field._values[index].compute()).item()


def require_scalar(field, operation):
    if field.dims:
        raise TypeError(f"{operation} currently requires scalar fields; field {field.name!r} contains arrays. "
                        "Select or extract suitable features explicitly; arrays are not flattened automatically")


def validate_ids(ids, length: int) -> tuple[str, ...]:
    if isinstance(ids, (str, bytes, da.Array)):
        raise TypeError("IDs must be an eager sequence of strings")
    result = tuple(ids)
    if len(result) != length:
        raise ValueError(f"Expected {length} sample IDs, got {len(result)}")
    seen = set()
    for value in result:
        if not isinstance(value, str) or not value:
            raise ValueError("Sample IDs must be non-empty strings")
        if value in seen:
            raise ValueError(f"Duplicate sample ID: {value!r}")
        seen.add(value)
    return result


def new_ids(length: int) -> tuple[str, ...]:
    return tuple(uuid4().hex for _ in range(length))


def infer_dtype(values) -> str:
    kinds = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_)):
            kinds.add("bool")
        elif isinstance(value, Integral):
            kinds.add("int64")
        elif isinstance(value, Real):
            kinds.add("float64")
        elif isinstance(value, str):
            kinds.add("string")
        else:
            raise TypeError("Fields currently support scalar strings, booleans, and numbers")
    if not kinds:
        if len(values):
            raise ValueError("Supply dtype for a field containing only missing values")
        return "float64"
    if kinds <= {"int64", "float64"}:
        return "float64" if "float64" in kinds else "int64"
    if len(kinds) != 1:
        raise TypeError("Mixed string, boolean, and numeric values require separate fields")
    return kinds.pop()


def validate_block(block, *, logical_dtype: str, nullable: bool, block_info=None):
    array = np.asarray(block)
    if array.ndim != 1:
        raise ValueError("Scalar fields require a one-dimensional array across samples")
    if block_info and tuple(array.shape) != tuple(block_info[None]["chunk-shape"]):
        raise ValueError("Lazy field values do not match their declared shape")
    result = []
    for value in array:
        if value is None:
            if not nullable:
                raise ValueError("Missing value in a non-nullable field")
            result.append(None)
            continue
        if logical_dtype == "string":
            valid = isinstance(value, str)
        elif logical_dtype == "bool":
            valid = isinstance(value, (bool, np.bool_))
        elif logical_dtype == "int64":
            valid = (isinstance(value, Integral)
                     and not isinstance(value, (bool, np.bool_))
                     and -(2**63) <= int(value) < 2**63)
        else:
            valid = isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            if valid:
                try:
                    valid = math.isfinite(float(value))
                except OverflowError:
                    valid = False
        if not valid:
            raise ValueError(f"Value {value!r} is incompatible with {logical_dtype}")
        result.append(value)
    storage_dtype = object if nullable or logical_dtype == "string" else np.dtype(logical_dtype)
    return np.array(result, dtype=storage_dtype)


def prepare_values(values, dtype, nullable):
    install_ipython_completion()
    if isinstance(values, da.Array):
        if values.ndim != 1 or not all(np.isfinite(values.shape)):
            raise ValueError("Scalar fields require one dimension with a known sample count; array values need explicit dims")
        if dtype is None:
            dtype = {"b": "bool", "i": "int64", "u": "int64",
                     "f": "float64", "U": "string"}.get(values.dtype.kind)
            if dtype is None:
                raise ValueError("Supply dtype for an object-backed Dask field")
        nullable = False if nullable is None else nullable
        storage_dtype = object if nullable or dtype == "string" else np.dtype(dtype)
        storage = values.copy().map_blocks(
            validate_block, logical_dtype=dtype, nullable=nullable,
            dtype=storage_dtype, meta=np.empty((0,), dtype=storage_dtype),
        )
        return storage, dtype, nullable
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, (Sequence, np.ndarray)):
        raise TypeError("Field values must be a sequence, NumPy array, or Dask array")
    array = np.array(values, dtype=object, copy=True)
    if array.ndim != 1:
        raise ValueError("Array-valued fields require explicit dims; construct a Field with dimension names")
    dtype = infer_dtype(array) if dtype is None else dtype
    nullable = any(value is None for value in array) if nullable is None else nullable
    array = validate_block(array, logical_dtype=dtype, nullable=nullable)
    array.flags.writeable = False
    return da.from_array(array, chunks=1024), dtype, nullable


def positions(selector, length: int) -> tuple[list[int], bool]:
    if isinstance(selector, (bool, np.bool_)):
        raise TypeError("Use an integer, slice, index sequence, or boolean mask")
    scalar = isinstance(selector, Integral)
    if scalar:
        result = [int(selector)]
    elif isinstance(selector, slice):
        result = list(range(length))[selector]
    else:
        if isinstance(selector, (str, bytes, da.Array)):
            raise TypeError("Sample selection must be an eager index sequence or boolean mask")
        try:
            result = list(selector)
        except TypeError:
            raise TypeError("Use an integer, slice, index sequence, or boolean mask") from None
        boolean_array = isinstance(selector, np.ndarray) and selector.dtype.kind == "b"
        if boolean_array or (result and all(isinstance(i, (bool, np.bool_)) for i in result)):
            if boolean_array and selector.ndim != 1:
                raise ValueError("Boolean masks must be one-dimensional")
            if len(result) != length:
                raise ValueError("Boolean mask length must match the sample count")
            result = [i for i, keep in enumerate(result) if keep]
    normalized = []
    for index in result:
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral):
            raise TypeError("Sample indices must be integers")
        index = int(index)
        index = index + length if index < 0 else index
        if not 0 <= index < length:
            raise IndexError("Sample index out of range")
        normalized.append(index)
    if len(set(normalized)) != len(normalized):
        raise ValueError("Selection cannot repeat sample IDs")
    return normalized, scalar


def clone_field(field, *, storage=None, ids=None, name=None):
    from ..data import Field, SampleField

    field_type = Field if isinstance(field, SampleField) else type(field)
    result = object.__new__(field_type)
    for attribute in FIELD_ATTRIBUTES:
        object.__setattr__(result, attribute, getattr(field, attribute))
    object.__setattr__(result, "_dataset", None)
    if storage is not None:
        object.__setattr__(result, "_values", storage)
    if ids is not None:
        object.__setattr__(result, "ids", ids)
    if name is not None:
        object.__setattr__(result, "name", name)
    return result


def bound_field(field, dataset):
    result = clone_field(field)
    object.__setattr__(result, "_dataset", dataset)
    return result


def dataset_fields(dataset):
    return MappingProxyType({name: bound_field(field, dataset) for name, field in dataset._fields.items()})


def initialise_sample_field(view, field, index):
    indices, scalar = positions(index, len(field))
    if not scalar:
        raise TypeError("A sample field requires one integer position")
    selected = take_field(field, indices)
    for attribute in FIELD_ATTRIBUTES:
        object.__setattr__(view, attribute, getattr(selected, attribute))


def take_field(field, indices):
    ids = tuple(field.ids[i] for i in indices)
    result = clone_field(field, storage=field._values[np.asarray(indices, dtype=np.intp)], ids=ids)
    object.__setattr__(result, "_dataset", field._dataset)
    if field._coordinate_fields:
        object.__setattr__(result, "_coordinate_fields", MappingProxyType(
            {dim: take_field(coordinate, indices) for dim, coordinate in field._coordinate_fields.items()}))
    return result


def align_field(field, ids, *, allow_extra=False):
    if field.ids == ids:
        return field
    available, required = set(field.ids), set(ids)
    if allow_extra and not required <= available:
        missing = [identity for identity in ids if identity not in available]
        raise ValueError(f"Supporting field sample IDs are missing required IDs: {missing[:5]!r}")
    if not allow_extra and available != required:
        raise ValueError("Assigned field sample IDs must match the dataset IDs")
    locations = {value: i for i, value in enumerate(field.ids)}
    return take_field(field, [locations[value] for value in ids])


def validate_names(fields):
    if not isinstance(fields, Mapping):
        raise TypeError("Dataset fields must be a mapping from names to values")
    if any(not isinstance(name, str) or not name for name in fields):
        raise ValueError("Field names must be non-empty strings")


def build_fields(fields, ids, field_type):
    validate_names(fields)
    fields = dict(fields)
    id_field = None
    if "id" in fields:
        if ids is not None:
            raise ValueError("Supply IDs through either the id field or ids, not both")
        supplied = fields.pop("id")
        if isinstance(supplied, field_type):
            if supplied.dtype != "string" or supplied.nullable or supplied.dims:
                raise ValueError("The id field must be a non-nullable string")
            id_field = supplied
        ids = supplied.values if isinstance(supplied, field_type) else supplied
    anchor = next((value for value in fields.values() if isinstance(value, field_type)), None)
    if ids is None and anchor is not None:
        ids = anchor.ids
    if ids is not None:
        if isinstance(ids, (str, bytes, da.Array)):
            raise TypeError("IDs must be an eager sequence of strings")
        ids = tuple(ids)
        length = len(ids)
    else:
        length = len(next(iter(fields.values()))) if fields else 0
        ids = new_ids(length)
    ids = validate_ids(ids, length)
    if id_field is not None:
        if id_field.ids != ids:
            raise ValueError("The id field's values must match its sample IDs")
    else:
        id_field = field_type(ids, ids=ids, dtype="string", label="Sample ID")
    result = {"id": id_field}
    for name, value in fields.items():
        field = align_field(value, ids) if isinstance(value, field_type) else field_type(value, ids=ids)
        result[name] = clone_field(field, name=name)
    result["id"] = clone_field(result["id"], name="id")
    from .tensor import bind_coordinates

    return MappingProxyType(bind_coordinates(result)), ids


def dataset_view(dataset, fields, ids):
    from .tensor import bind_coordinates

    result = object.__new__(type(dataset))
    fields = {name: clone_field(field) for name, field in fields.items()}
    object.__setattr__(result, "_fields", MappingProxyType(bind_coordinates(fields)))
    object.__setattr__(result, "_ids", ids)
    return result


def update_dataset_metadata(dataset, fields, updates):
    if fields is not None and not isinstance(fields, Mapping):
        raise TypeError("Field updates must be supplied as a mapping")
    patches = {} if fields is None else dict(fields)
    if patches.keys() & updates.keys():
        raise ValueError("A field was supplied both in the mapping and as a keyword")
    patches.update(updates)
    validate_names(patches)
    for name in patches:
        if name not in dataset.fields:
            raise KeyError(f"No field named {name!r}; update only changes existing fields")
    result = dict(dataset.fields)
    for name, changes in patches.items():
        try:
            metadata = updated_metadata(dataset.fields[name], changes)
        except (TypeError, ValueError) as error:
            raise type(error)(f"Field {name!r}: {error}") from error
        if not changes:
            continue
        field = clone_field(dataset.fields[name])
        for attribute, value in metadata.items():
            object.__setattr__(field, attribute, value)
        result[name] = field
    return dataset_view(dataset, result, dataset.ids)


def lookup_field(fields, name):
    try:
        return fields[name]
    except KeyError:
        raise AttributeError(f"No field or attribute named {name!r}; use brackets for field names") from None


def dataset_dependencies(data):
    from dask.base import tokenize

    from .tensor import CoordinateReference

    fields, signatures, visiting = {}, {}, set()

    def visit(name, field):
        if name in visiting:
            raise ValueError("Coordinate references cannot form cycles")
        signature = tokenize(field._values, field.ids, field.dtype, field.dims,
                             field._shape, field.nullable, dict(field._coords),
                             {key: getattr(field, key) for key in DISPLAY_ATTRIBUTES}, field.prediction)
        if name in fields:
            if signatures[name] != signature:
                raise ValueError(f"Ambiguous coordinate reference {name!r}; expose one shared coordinate field first")
            return
        if field.ids != data.ids:
            raise ValueError(f"Field {name!r} does not match the dataset sample IDs")
        visiting.add(name)
        fields[name], signatures[name] = field, signature
        for dim, coordinate in field._coords.items():
            if isinstance(coordinate, CoordinateReference):
                if dim not in field._coordinate_fields:
                    raise ValueError(f"Field {name!r}: unresolved coordinate {dim!r}")
                visit(coordinate.name, field._coordinate_fields[dim])
        visiting.remove(name)

    for name, field in data._fields.items():
        visit(name, field)
    return fields
