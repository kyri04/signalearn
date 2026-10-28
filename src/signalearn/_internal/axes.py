from numbers import Integral, Real
from types import MappingProxyType

from dask.base import tokenize
import dask.array as da
import numpy as np

from .data import clone_field, dataset_view, positions
from .metadata import DISPLAY_ATTRIBUTES
from .preprocess import selection_source
from .tensor import ArrayStorage, CoordinateReference, IndexedArray, batch_task, check_coordinates

WINDOW_FIELDS = ("source_id", "window_index", "window_start", "window_stop")


def same_coordinate(left, right):
    return (left.ids == right.ids and left.dtype == right.dtype and left.dims == right.dims
            and left._coords == right._coords and left.prediction == right.prediction
            and all(getattr(left, key) == getattr(right, key) for key in DISPLAY_ATTRIBUTES)
            and (left._values is right._values or tokenize(left._values) == tokenize(right._values)))


def axis_context(x, dim):
    data = selection_source(x)
    if not isinstance(dim, str) or dim not in x.dims:
        raise ValueError(f"dim must identify one of the field's named dimensions: {x.dims!r}")
    if dim == "class" and x.prediction is not None:
        raise ValueError("Changing a prediction class axis requires an explicit prediction conversion")
    nodes = {}

    def visit(name, field):
        if name in nodes:
            if not same_coordinate(nodes[name], field):
                raise ValueError(f"Ambiguous coordinate reference {name!r}; expose one shared coordinate field first")
            return
        nodes[name] = field
        for axis, specification in field._coords.items():
            if isinstance(specification, CoordinateReference):
                visit(specification.name, field._coordinate_fields[axis])

    for name, field in data._fields.items():
        visit(name, field)
    if x.name not in nodes or not same_coordinate(nodes[x.name], x):
        raise ValueError("The selected field is incompatible with its current dataset context")
    affected, pending = {}, [(x.name, x.dims.index(dim))]
    while pending:
        name, axis = pending.pop()
        if axis in affected.setdefault(name, set()):
            continue
        affected[name].add(axis)
        field = nodes[name]
        coordinate = field._coords.get(field.dims[axis])
        if isinstance(coordinate, CoordinateReference):
            pending.append((coordinate.name, 0))
        if len(field.dims) == 1:
            for other_name, other in nodes.items():
                for other_dim, specification in other._coords.items():
                    if isinstance(specification, CoordinateReference) and specification.name == name:
                        pending.append((other_name, other.dims.index(other_dim)))
    return data, nodes, affected


def axis_positions(indices, length):
    try:
        selected, _ = positions(indices, length)
    except ValueError as error:
        if "repeat sample IDs" in str(error):
            raise ValueError("Axis selections cannot repeat positions") from None
        raise
    return tuple(selected)


def positional_plan(x, dim, indices):
    if not isinstance(indices, (Integral, slice, tuple, list, range, np.ndarray)):
        raise TypeError("indices must be an integer, slice, or eager index sequence")
    axis = x.dims.index(dim)
    cache, selected = {}, []
    for item in x._values.items:
        length = item.shape[axis]
        if length not in cache:
            cache[length] = axis_positions(indices, length)
        selected.append(cache[length])
    hint = x._values.sample_shape or x._shape
    default = axis_positions(indices, hint[axis]) if hint is not None and hint[axis] is not None else None
    return selected, default


def numeric_coordinate(x, dim, nodes):
    specification = x._coords.get(dim)
    if isinstance(specification, CoordinateReference):
        field = nodes[specification.name]
        values = batch_task(field, tuple(range(len(field)))).compute()
        unit, unit_latex, dtype = field.unit, field.unit_latex, field.dtype
        default = None
    elif specification is not None:
        default = np.array(specification)
        values = (default,) * len(x)
        unit, unit_latex, dtype = None, None, default.dtype.name
    elif len(x.dims) == 1 and any(isinstance(value, CoordinateReference) and value.name == x.name
                                 for field in nodes.values() for value in field._coords.values()):
        values = batch_task(nodes[x.name], tuple(range(len(x)))).compute()
        unit, unit_latex, dtype = x.unit, x.unit_latex, x.dtype
        default = None
    else:
        raise ValueError(f"Dimension {dim!r} requires explicit numeric coordinates for this operation")
    if np.dtype(dtype).kind not in "iuf":
        raise TypeError("Coordinate intervals require numeric coordinates")
    for value in values:
        if value.ndim != 1 or not np.isfinite(value).all():
            raise ValueError("Coordinates must be one-dimensional, finite, and complete")
    return values, unit, unit_latex, dtype, default


def coordinate_bounds(start, stop):
    if start is None and stop is None:
        raise ValueError("Supply at least one coordinate bound")
    for value in (start, stop):
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError("Coordinate bounds must be finite numbers")
            if not isinstance(value, Integral) and not np.isfinite(value):
                raise ValueError("Coordinate bounds must be finite numbers")
    if start is not None and stop is not None and start > stop:
        raise ValueError("start must not exceed stop")


def interval_positions(values, start, stop):
    return tuple(index for index, value in enumerate(values.tolist())
                 if (start is None or value >= start) and (stop is None or value <= stop))


def coordinate_name(name, dim, nodes):
    prefix = f"{name}_{dim}_coordinate"
    candidate, suffix = prefix, 1
    while candidate in nodes:
        candidate = f"{prefix}_{suffix}"
        suffix += 1
    return candidate


def selected_storage(field, axes, rows, selections, default):
    items = []
    for index, source in enumerate(rows):
        item = field._values.items[source]
        if axes:
            shape = tuple(len(selections[index]) if axis in axes else size for axis, size in enumerate(item.shape))
            indices = tuple((axis, selections[index]) for axis in sorted(axes))
            item = IndexedArray(item, indices, shape, item.dtype)
        items.append(item)
    hint = field._values.sample_shape
    if items:
        hint = tuple(items[0].shape) if all(item.shape == items[0].shape for item in items) else None
    elif axes:
        hint = tuple(len(default) if axis in axes else size for axis, size in enumerate(hint)) if hint is not None and default is not None else None
    return ArrayStorage(tuple(items), field._values.dtype, hint)


def selected_coordinates(field, name, axes, selections, default, ids, nodes):
    from ..data import Field

    coordinates, additions = dict(field._coords), {}
    for axis in axes:
        dim = field.dims[axis]
        value = coordinates.get(dim)
        if value is None or isinstance(value, CoordinateReference):
            continue
        selected = [tuple(value[index] for index in selection) for selection in selections]
        if not selected:
            coordinates[dim] = tuple(value[index] for index in default) if default is not None else value
        elif all(item == selected[0] for item in selected):
            coordinates[dim] = selected[0]
        else:
            values = np.array(value)
            if values.dtype.kind not in "iuf":
                raise ValueError("Different per-sample string coordinates require an explicit supported representation")
            key = coordinate_name(name, dim, nodes | additions)
            coordinate = Field([np.take(values, selection) for selection in selections], ids=ids,
                               dims=(dim,), dtype=values.dtype.name, label=dim)
            additions[key] = clone_field(coordinate, name=key)
            coordinates[dim] = CoordinateReference(key)
    return coordinates, additions


def axis_dataset(data, nodes, affected, rows, selections, ids, default=None):
    output = {}
    for name, field in nodes.items():
        axes = affected.get(name, set())
        if field.dims:
            storage = selected_storage(field, axes, rows, selections, default)
        else:
            values = np.asarray(ids, dtype=object) if name == "id" else field._values[np.asarray(rows, dtype=np.intp)]
            storage = da.from_array(values, chunks=1024) if name == "id" else values
        result = clone_field(field, storage=storage, ids=ids, name=name)
        if axes:
            shape = list(field._shape or (None,) * len(field.dims))
            for axis in axes:
                lengths = [item.shape[axis] for item in storage.items]
                shape[axis] = lengths[0] if lengths and len(set(lengths)) == 1 else len(default) if not lengths and default is not None else None
            object.__setattr__(result, "_shape", tuple(shape))
            coordinates, additions = selected_coordinates(field, name, axes, selections, default, ids, nodes | output)
            object.__setattr__(result, "_coords", MappingProxyType(coordinates))
            output.update(additions)
        output[name] = result
    for field in output.values():
        bindings = {dim: output[value.name] for dim, value in field._coords.items() if isinstance(value, CoordinateReference)}
        object.__setattr__(field, "_coordinate_fields", MappingProxyType(bindings))
        if field.dims:
            check_coordinates(field)
    return dataset_view(data, {name: output[name] for name in data._fields}, ids)


def window_parameters(data, size, step, allow_partial):
    step = size if step is None else step
    for name, value in (("size", size), ("step", step)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number in coordinate units")
    if not isinstance(allow_partial, bool):
        raise TypeError("allow_partial must be a boolean")
    collisions = set(WINDOW_FIELDS) & set(data._fields)
    if collisions:
        raise ValueError(f"Window provenance fields already exist: {sorted(collisions)!r}; they cannot be overwritten")
    return step


def interval_count(value, spacing, name):
    ratio = value / spacing
    if not np.isfinite(ratio):
        raise ValueError(f"{name} is outside the supported coordinate range")
    count = round(ratio)
    if count < 1 or not np.isclose(ratio, float(count), rtol=1e-7, atol=1e-9):
        raise ValueError(f"{name} must be a positive whole multiple of the coordinate sampling interval")
    return count


def window_plan(coordinates, size, step, allow_partial):
    rows, selections, starts, stops, numbers = [], [], [], [], []
    for source, values in enumerate(coordinates):
        if not len(values):
            continue
        if len(values) < 2:
            raise ValueError("Windowing requires at least two coordinate values to establish regular spacing")
        native = values.tolist()
        shifted = np.array([value - native[0] for value in native], dtype=float)
        spacing = shifted[-1] / (len(values) - 1)
        tolerance = np.finfo(values.dtype).eps * max(abs(value) for value in shifted) if values.dtype.kind == "f" else 0
        expected = np.arange(len(values), dtype=float) * spacing
        if (not np.isfinite(spacing) or spacing <= 0 or not (np.diff(shifted) > 0).all()
                or not np.allclose(shifted, expected, rtol=1e-12, atol=tolerance)):
            raise ValueError("Window coordinates must be strictly increasing and regularly spaced; resample explicitly first")
        count, stride = interval_count(size, spacing, "size"), interval_count(step, spacing, "step")
        for number, begin in enumerate(range(0, len(values), stride)):
            end = begin + count
            if end > len(values) and not allow_partial:
                break
            end = min(end, len(values))
            boundary = native[end] if end < len(values) else native[-1] + (int(round(spacing)) if values.dtype.kind in "iu" else spacing)
            rows.append(source)
            selections.append(tuple(range(begin, end)))
            starts.append(native[begin])
            stops.append(boundary)
            numbers.append(number)
    return rows, selections, starts, stops, numbers


def window_metadata(data, rows, starts, stops, numbers, ids, unit, unit_latex, dtype):
    from ..data import Field

    kind = "int64" if np.dtype(dtype).kind in "iu" else "float64"
    return {
        "source_id": Field([data.ids[index] for index in rows], ids=ids, dtype="string", label="Source sample ID"),
        "window_index": Field(numbers, ids=ids, dtype="int64", label="Window index"),
        "window_start": Field(starts, ids=ids, dtype=kind, unit=unit, unit_latex=unit_latex, label="Window start"),
        "window_stop": Field(stops, ids=ids, dtype=kind, unit=unit, unit_latex=unit_latex, label="Window stop"),
    }
