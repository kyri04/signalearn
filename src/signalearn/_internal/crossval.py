from copy import deepcopy

from dask import compute, delayed
import dask.array as da
import numpy as np

from ..data import Dataset, Field
from .data import align_field, clone_field, dataset_dependencies, dataset_view, new_ids, require_scalar
from .tensor import ArrayStorage, batch_task


def crossval_inputs(splits, procedure, target):
    from ..preprocess import Folds

    if not isinstance(splits, Folds):
        raise TypeError("crossval requires a Folds plan from sl.folds")
    if not callable(procedure):
        raise TypeError("procedure must be callable as procedure(train, test)")
    if target is not None:
        if not isinstance(target, Field):
            raise TypeError("target must be a SignaLearn Field")
        require_scalar(target, "Cross-validation reference target")
        target = align_field(target, splits.data.ids, allow_extra=True)
    rows = tuple(index for _, test in splits.indices for index in test)
    repeated = len(set(rows)) != len(rows)
    ids = new_ids(len(rows)) if repeated else tuple(splits.data.ids[index] for index in rows)
    return deepcopy(procedure), target, rows, ids, repeated


def prediction_output(output, test):
    if not isinstance(output, Dataset):
        raise TypeError("procedure must return a prediction Dataset")
    if len(output._fields) < 2:
        raise ValueError("procedure must return at least one prediction field")
    if set(output.ids) != set(test.ids):
        raise ValueError("Prediction sample IDs must match the held-out test IDs exactly")
    locations = {identity: index for index, identity in enumerate(output.ids)}
    output = output[[locations[identity] for identity in test.ids]]
    if {"fold", "source_id"} & dataset_dependencies(output).keys():
        raise ValueError("Prediction fields and coordinates cannot use reserved names 'fold' or 'source_id'")
    return output


def reference_output(output, target, indices):
    if target is None:
        return output
    base = f"{target.name or 'target'}_target"
    name, suffix = base, 2
    names = dataset_dependencies(output)
    while name in names or name in {"fold", "source_id"}:
        name, suffix = f"{base}_{suffix}", suffix + 1
    return output.assign({name: target[list(indices)]})


def frozen_values(values):
    result = np.array(values, copy=True)
    result.flags.writeable = False
    return result


def snapshot_predictions(output, ids, fold):
    fields = dataset_dependencies(output)
    tasks = [batch_task(field, tuple(range(len(output)))) if field.dims else delayed(field._values)
             for field in fields.values()]
    batches = compute(*tasks)
    frozen = {}
    for (name, field), values in zip(fields.items(), batches):
        storage = (ArrayStorage(tuple(frozen_values(value) for value in values),
                                np.dtype(field.dtype), field._values.sample_shape) if field.dims
                   else da.from_array(frozen_values(values), chunks=1024))
        frozen[name] = clone_field(field, storage=storage, ids=ids)
    frozen["id"] = clone_field(fields["id"], storage=da.from_array(frozen_values(ids), chunks=1024), ids=ids)
    result = dataset_view(output, frozen, ids).select(*output._fields)
    return result.assign(
        source_id=Field(output.ids, ids=ids, dtype="string", label="Source sample ID"),
        fold=Field([fold] * len(ids), ids=ids, dtype="int64", label="Fold"),
    )
