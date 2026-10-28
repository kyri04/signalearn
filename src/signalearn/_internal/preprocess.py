from copy import deepcopy
from collections.abc import Sequence
from numbers import Integral, Real
from types import MappingProxyType

from dask import delayed
import dask.array as da
import numpy as np

from ..data import Dataset, Field
from .data import align_field, clone_field, materialise_values, require_scalar, validate_block, validate_ids
from .metadata import DISPLAY_ATTRIBUTES
from .tensor import ArrayStorage, CoordinateReference, batch_task, sample_task


def fold_vector(values, length):
    if isinstance(values, da.Array):
        raise TypeError("Split indices must be eager integer vectors")
    values = np.asarray(values)
    if values.ndim != 1 or not values.size:
        raise ValueError("Training and test indices must be non-empty one-dimensional vectors")
    if values.dtype.kind not in "iu":
        raise TypeError("Split indices must be integers")
    indices = tuple(int(value) for value in values)
    if any(value < 0 or value >= length for value in indices):
        raise ValueError("Split indices are outside the dataset")
    if len(set(indices)) != len(indices):
        raise ValueError("A split cannot repeat indices within a partition")
    return indices


def validate_folds(data, indices):
    if not isinstance(data, Dataset):
        raise TypeError("folds requires a Dataset")
    if not len(data):
        raise ValueError("folds requires at least one sample")
    result = []
    for pair in indices:
        if not isinstance(pair, (tuple, list, np.ndarray)) or len(pair) != 2:
            raise ValueError("Each split must contain training and test index vectors")
        train, test = (fold_vector(values, len(data)) for values in pair)
        if set(train) & set(test):
            raise ValueError("Training and test indices must be disjoint within each split")
        result.append((train, test))
    if not result:
        raise ValueError("At least one split is required")
    return tuple(result)


def fold_fields(data, target, group):
    if not isinstance(data, Dataset):
        raise TypeError("folds requires a Dataset")
    if not len(data):
        raise ValueError("folds requires at least one sample")
    fields = []
    for name, field in (("target", target), ("group", group)):
        if field is not None:
            if not isinstance(field, Field):
                raise TypeError(f"{name} must be a SignaLearn Field")
            require_scalar(field, f"Fold {name}")
            field = align_field(field, data.ids)
        fields.append(field)
    return tuple(fields)


def fold_columns(target, group):
    from dask import compute

    from .alignment import native_column

    fields = [field for field in (target, group) if field is not None]
    arrays = iter(compute(*(field._values for field in fields)))
    columns = []
    for name, field in (("target", target), ("group", group)):
        if field is None:
            columns.append(None)
            continue
        values = next(arrays)
        if any(value is None for value in values):
            raise ValueError(f"Fold {name} values must not be missing")
        columns.append(native_column(values, field.dtype))
    return tuple(columns)


def validate_fold_groups(indices, group):
    if group is None:
        return
    for train, test in indices:
        if {group[index] for index in train} & {group[index] for index in test}:
            raise ValueError("The splitter placed members of the supplied group in both training and test partitions")


def validate_split(data, sizes, group, seed, shuffle):
    if not isinstance(data, Dataset):
        raise TypeError("data must be a SignaLearn Dataset")
    if isinstance(sizes, (str, bytes)) or not isinstance(sizes, (Sequence, np.ndarray)):
        raise TypeError("sizes must be a sequence of proportions")
    sizes = tuple(sizes)
    if len(sizes) < 2:
        raise ValueError("Supply at least two split proportions")
    if any(isinstance(size, (bool, np.bool_)) or not isinstance(size, Real)
           or not np.isfinite(size) or size <= 0 for size in sizes):
        raise ValueError("Split proportions must be finite positive numbers")
    if not np.isclose(sum(sizes), 1, rtol=0, atol=1e-12):
        raise ValueError("Split proportions must sum to one")
    if not isinstance(shuffle, bool):
        raise TypeError("shuffle must be a boolean")
    if seed is not None:
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a non-negative integer or None")
        if not shuffle:
            raise ValueError("seed is only applicable when shuffle=True")
    if group is not None and not isinstance(group, Field):
        raise TypeError("group must be a SignaLearn Field")
    if group is not None:
        require_scalar(group, "Split grouping")
    return np.asarray(sizes, dtype=float)


def split_units(data, group):
    if group is None:
        return np.arange(len(data)), len(data)
    values = materialise_values(align_field(group, data.ids))
    if any(value is None for value in values):
        raise ValueError("Split grouping values must not be missing")
    categories = {value: index for index, value in enumerate(dict.fromkeys(values))}
    return np.array([categories[value] for value in values], dtype=np.intp), len(categories)


def partition_indices(units, count, sizes, seed, shuffle):
    exact = sizes * count
    counts = np.floor(exact).astype(np.intp)
    remainder = count - int(counts.sum())
    order = np.argsort(-(exact - counts), kind="stable")
    counts[order[:remainder]] += 1
    if (counts == 0).any():
        raise ValueError("Split proportions would produce an empty partition; use more samples or groups")
    order = np.arange(count)
    if shuffle:
        order = np.random.default_rng(None if seed is None else int(seed)).permutation(order)
    partitions = np.split(order, np.cumsum(counts)[:-1])
    return [np.flatnonzero(np.isin(units, selected)) for selected in partitions]


def deferred_transform(adapter, model, inputs, shape):
    component = adapter.snapshot(model)
    return delayed(adapter.apply_transform)(component, inputs[0], inputs[2], shape)


def component_values(field, index):
    if field.dims:
        return batch_task(field, tuple(range(len(field)))) if index is None else sample_task(field, index)
    return field._values if index is None else delayed(component_input_scalar)(field._values[index])


def component_input_scalar(value):
    return np.asarray(value).item()


def component_tasks(adapter, model, fields, names, parameters, ids, schema):
    from ..data import Field

    options = {name: align_field(value, ids) if isinstance(value, Field) else deepcopy(value)
               for name, value in parameters.items()}
    batch = model.signalearn_mode == "batch"
    indices = (None,) if batch else range(len(ids))
    runtime_schema = {name: {"dtype": specification["dtype"], "nullable": specification["nullable"],
                             "dims": specification["attributes"]["dims"]}
                      for name, specification in schema.items()}
    tasks = []
    for index in indices:
        values = {name: component_values(field, index) for name, field in zip(names, fields)}
        arguments = {name: component_values(value, index) if isinstance(value, Field) else value
                     for name, value in options.items()}
        shapes = {name: specification["shapes"] if batch else specification["shapes"][index]
                  for name, specification in schema.items()}
        tasks.append(delayed(adapter.apply_transform)(model, values, arguments,
                     delayed(runtime_schema, traverse=False), delayed(shapes, traverse=False), batch))
    return tasks


def component_column(result, name, index=None):
    value = result[name]
    return value if index is None else value[index]


def component_scalar(result, name):
    return np.asarray([result[name]], dtype=object)


def component_dataset(tasks, schema, fields, names, ids, batch):
    from ..data import Dataset, Field

    outputs = {}
    inputs = dict(zip(names, fields))
    for name, specification in schema.items():
        attributes = specification["attributes"]
        dtype = np.dtype(specification["dtype"]) if specification["dtype"] != "string" else object
        if attributes["dims"]:
            items = tuple(da.from_delayed(delayed(component_column)(tasks[0] if batch else tasks[index], name,
                                        index if batch else None), shape=shape, dtype=dtype)
                          for index, shape in enumerate(specification["shapes"]))
            storage = ArrayStorage(items, np.dtype(dtype))
        elif batch:
            storage = da.from_delayed(delayed(component_column)(tasks[0], name), shape=(len(ids),), dtype=object)
        else:
            storage = da.concatenate([da.from_delayed(delayed(component_scalar)(task, name), shape=(1,), dtype=object)
                                      for task in tasks])
        field = Field(storage, ids=ids, **attributes)
        source = specification["source"]
        bindings = dict(source._coordinate_fields) if source is not None else {}
        for dim, coordinate in field._coords.items():
            if isinstance(coordinate, CoordinateReference) and coordinate.name in inputs and coordinate.name not in schema:
                bindings[dim] = inputs[coordinate.name]
        object.__setattr__(field, "_coordinate_fields", MappingProxyType(bindings))
        outputs[name] = field
    return Dataset(outputs, ids=ids)


def selection_source(x):
    from ..data import Field

    if not isinstance(x, Field):
        raise TypeError("Selection requires a SignaLearn Field accessed from a Dataset")
    dataset = x._dataset
    if dataset is None:
        raise ValueError("This field has no dataset context; access a field from a Dataset before selecting samples")
    if x.ids == dataset.ids:
        return dataset
    locations = {identity: index for index, identity in enumerate(dataset.ids)}
    if any(identity not in locations for identity in x.ids):
        raise ValueError("Field sample IDs are incompatible with its dataset context")
    return dataset[[locations[identity] for identity in x.ids]]


def matching_values(x, value):
    require_scalar(x, "filter")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (list, tuple, set, frozenset, np.ndarray)):
        if isinstance(value, np.ndarray) and value.ndim != 1:
            raise ValueError("Filter values must be scalars or a one-dimensional collection")
        values = list(value)
    else:
        values = [value]
    validate_block(values, logical_dtype=x.dtype, nullable=True)
    return frozenset(values)


def range_bounds(x, lower, upper):
    require_scalar(x, "between")
    if x.dtype not in {"int64", "float64"}:
        raise TypeError("between requires a numeric scalar field")
    if lower is None and upper is None:
        raise ValueError("Supply at least one range bound")
    for bound in (lower, upper):
        if bound is not None:
            if isinstance(bound, (bool, np.bool_)) or not isinstance(bound, Real):
                raise TypeError("Range bounds must be finite numbers")
            if not isinstance(bound, Integral) and not np.isfinite(bound):
                raise ValueError("Range bounds must be finite numbers")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("lower must not exceed upper")


def predicate_mask(values, condition):
    result = []
    for value in values:
        matched = condition(value)
        if not isinstance(matched, (bool, np.bool_)):
            raise TypeError("where conditions must return one boolean per sample")
        result.append(bool(matched))
    return result


def requested_ids(x, ids):
    from ..data import Field

    if x.name != "id" or x.dtype != "string" or x.nullable or x.dims:
        raise ValueError("take requires the dataset's id field")
    if isinstance(ids, str):
        ids = [ids]
    elif isinstance(ids, Field):
        require_scalar(ids, "Requested IDs")
        if ids.dtype != "string":
            raise TypeError("Requested IDs must be strings")
        ids = materialise_values(ids).tolist()
    elif not isinstance(ids, (Sequence, np.ndarray)) or isinstance(ids, bytes):
        raise TypeError("Requested IDs must be a string, an eager sequence, or a string Field")
    identities = validate_ids(ids, len(ids))
    locations = {identity: index for index, identity in enumerate(x.ids)}
    missing = [identity for identity in identities if identity not in locations]
    if missing:
        raise KeyError(f"Unknown sample IDs in this dataset view: {missing!r}")
    return [locations[identity] for identity in identities]


def sample_indices(data, n, seed):
    from ..data import Dataset

    if not isinstance(data, Dataset):
        raise TypeError("sample requires a Dataset")
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < 0:
        raise ValueError("n must be a non-negative integer")
    if n > len(data):
        raise ValueError("Cannot sample more observations than the dataset contains")
    if seed is not None and (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0):
        raise ValueError("seed must be a non-negative integer or None")
    return np.sort(np.random.default_rng(seed).choice(len(data), size=int(n), replace=False))


def concat_inputs(datasets):
    from ..data import Dataset

    if not isinstance(datasets, Sequence) or isinstance(datasets, (str, bytes)) or not datasets:
        raise TypeError("concat requires a non-empty sequence of Datasets")
    if any(not isinstance(dataset, Dataset) for dataset in datasets):
        raise TypeError("concat inputs must all be Datasets")
    names = tuple(datasets[0]._fields)
    if any(set(dataset._fields) != set(names) for dataset in datasets):
        raise ValueError("Concatenated datasets must have exactly the same field names")
    ids = tuple(identity for dataset in datasets for identity in dataset.ids)
    validate_ids(ids, len(ids))
    return names, ids


def concat_metadata(fields, name):
    result = {}
    for attribute in DISPLAY_ATTRIBUTES:
        values = [getattr(field, attribute) for field in fields]
        if attribute in {"colours", "markers"}:
            merged = {}
            for mapping in values:
                for category, setting in mapping.items():
                    if category in merged and merged[category] != setting:
                        raise ValueError(f"Field {name!r}: conflicting {attribute} for category {category!r}")
                    merged[category] = setting
            result[attribute] = MappingProxyType(merged)
        elif attribute == "unit":
            if any(value != values[0] for value in values):
                raise ValueError(f"Field {name!r}: units must match exactly; no conversion is inferred")
            result[attribute] = values[0]
        else:
            known = [value for value in values if value is not None]
            if known and any(value != known[0] for value in known):
                raise ValueError(f"Field {name!r}: conflicting {attribute} metadata")
            result[attribute] = known[0] if known else None
    return result


def concatenated_field(fields, name, ids):
    from .prediction import aligned_prediction_fields

    fields = aligned_prediction_fields(fields, name)
    first = fields[0]
    if any(field.dtype != first.dtype or field.dims != first.dims for field in fields):
        raise ValueError(f"Field {name!r}: dtypes and named dimensions must match exactly")
    if any(field._coords != first._coords for field in fields):
        raise ValueError(f"Field {name!r}: coordinate declarations must match; prepare compatible coordinates explicitly")
    metadata = concat_metadata(fields, name)
    if first.dims:
        items = tuple(item for field in fields for item in field._values.items)
        hints = [field._values.sample_shape for field in fields]
        hint = hints[0] if all(value == hints[0] for value in hints) else None
        if items:
            hint = tuple(items[0].shape) if all(item.shape == items[0].shape for item in items) else None
        storage = ArrayStorage(items, first._values.dtype, hint)
    else:
        storage = da.concatenate([field._values for field in fields])
    result = clone_field(first, storage=storage, ids=ids, name=name)
    object.__setattr__(result, "nullable", any(field.nullable for field in fields))
    if first.dims:
        constraints = [field._shape or (None,) * len(first.dims) for field in fields]
        shape = tuple(sizes[0] if all(size == sizes[0] for size in sizes) else None for sizes in zip(*constraints))
        object.__setattr__(result, "_shape", shape)
        bindings = {dim: concatenated_field([field._coordinate_fields[dim] for field in fields], coordinate.name, ids)
                    for dim, coordinate in first._coords.items() if isinstance(coordinate, CoordinateReference)}
        object.__setattr__(result, "_coordinate_fields", MappingProxyType(bindings))
    for attribute, value in metadata.items():
        object.__setattr__(result, attribute, value)
    return result
