from builtins import slice as python_slice
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ._internal.adapters import get_adapter
from ._internal.adapters.splitting import native_splits
from ._internal.alignment import model_inputs, output_dataset, selected_fields
from ._internal.axes import (
    axis_context, axis_dataset, coordinate_bounds, interval_positions, numeric_coordinate,
    positional_plan, window_metadata, window_parameters, window_plan,
)
from ._internal.data import dataset_view, materialise_values, new_ids, require_scalar
from ._internal.preprocess import (
    component_dataset, component_tasks, concat_inputs, concatenated_field, deferred_transform,
    fold_columns, fold_fields, matching_values, partition_indices, predicate_mask, range_bounds, requested_ids,
    sample_indices, selection_source, split_units, validate_fold_groups, validate_folds, validate_split,
)
from .data import Dataset, Field

__all__ = ["Folds", "folds", "transform", "split", "filter", "between", "where", "sort", "take", "sample", "concat",
           "slice", "crop", "window"]


@dataclass(frozen=True, eq=False)
class Folds(Sequence):
    data: Dataset
    indices: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]

    def __post_init__(self):
        object.__setattr__(self, "indices", validate_folds(self.data, self.indices))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, python_slice):
            return tuple((self.data[list(train)], self.data[list(test)]) for train, test in self.indices[index])
        train, test = self.indices[index]
        return self.data[list(train)], self.data[list(test)]

    def __repr__(self):
        return f"Folds(splits={len(self)}, samples={len(self.data)})"


def folds(
    data: Dataset, splitter: Any, *, target: Field | None = None, group: Field | None = None,
) -> Folds:
    target, group = fold_fields(data, target, group)
    labels, groups = fold_columns(target, group)
    plan = Folds(data, native_splits(splitter, len(data), labels, groups))
    validate_fold_groups(plan.indices, groups)
    return plan


def transform(
    x: Field | Sequence[Field] | Dataset, model: Any,
    **transform_params: Any,
) -> Dataset:
    adapter = get_adapter(model, "transform")
    if getattr(adapter, "structured", False):
        model = adapter.snapshot(model)
    fields, names, ids = selected_fields(x)
    fields, names = adapter.fitted_inputs(model, fields, names)
    schema = adapter.output_schema(model, fields, names)
    if getattr(adapter, "structured", False):
        tasks = component_tasks(adapter, model, fields, names, transform_params, ids, schema)
        return component_dataset(tasks, schema, fields, names, ids, model.signalearn_mode == "batch")
    inputs = model_inputs(fields, None, transform_params, ids, snapshot=True)
    result = deferred_transform(adapter, model, inputs, (len(ids), len(schema)))
    return output_dataset(result, schema, ids)


def split(
    data: Dataset, sizes: Sequence[float] = (0.8, 0.2), *,
    group: Field | None = None, seed: int | None = None, shuffle: bool = True,
) -> tuple[Dataset, ...]:
    proportions = validate_split(data, sizes, group, seed, shuffle)
    units, count = split_units(data, group)
    indices = partition_indices(units, count, proportions, seed, shuffle)
    return tuple(data[selection] for selection in indices)


def filter(x: Field, value: Any) -> Dataset:
    data = selection_source(x)
    wanted = matching_values(x, value)
    if not wanted:
        return data[:0]
    values = materialise_values(x).tolist()
    return data[[index for index, observed in enumerate(values) if observed in wanted]]


def between(x: Field, lower: int | float | None = None, upper: int | float | None = None) -> Dataset:
    data = selection_source(x)
    range_bounds(x, lower, upper)
    values = materialise_values(x).tolist()
    return data[[index for index, value in enumerate(values) if value is not None
                 and (lower is None or value >= lower) and (upper is None or value <= upper)]]


def where(x: Field, condition: Callable[[Any], bool]) -> Dataset:
    data = selection_source(x)
    if not callable(condition):
        raise TypeError("condition must be callable")
    values = materialise_values(x)
    if not x.dims:
        values = values.tolist()
    return data[predicate_mask(values, condition)]


def sort(x: Field, descending: bool = False) -> Dataset:
    data = selection_source(x)
    require_scalar(x, "sort")
    if not isinstance(descending, bool):
        raise TypeError("descending must be a boolean")
    values = materialise_values(x)
    present = [index for index, value in enumerate(values) if value is not None]
    ordered = sorted(present, key=lambda index: values[index], reverse=descending)
    ordered.extend(index for index, value in enumerate(values) if value is None)
    return data[ordered]


def take(x: Field, ids: str | Sequence[str] | Field) -> Dataset:
    data = selection_source(x)
    indices = requested_ids(x, ids)
    return data[indices]


def sample(data: Dataset, n: int, seed: int | None = None) -> Dataset:
    indices = sample_indices(data, n, seed)
    return data[indices]


def concat(datasets: Sequence[Dataset]) -> Dataset:
    names, ids = concat_inputs(datasets)
    fields = {name: concatenated_field([dataset._fields[name] for dataset in datasets], name, ids)
              for name in names}
    return dataset_view(datasets[0], fields, ids)


def slice(x: Field, dim: str, indices: Any) -> Dataset:
    data, nodes, affected = axis_context(x, dim)
    selections, default = positional_plan(x, dim, indices)
    return axis_dataset(data, nodes, affected, range(len(data)), selections, data.ids, default)


def crop(x: Field, dim: str, start: int | float | None = None, stop: int | float | None = None) -> Dataset:
    data, nodes, affected = axis_context(x, dim)
    coordinate_bounds(start, stop)
    coordinates, _, _, _, default = numeric_coordinate(x, dim, nodes)
    selections = [interval_positions(values, start, stop) for values in coordinates]
    default = interval_positions(default, start, stop) if default is not None else None
    return axis_dataset(data, nodes, affected, range(len(data)), selections, data.ids, default)


def window(
    x: Field, dim: str, size: int | float, step: int | float | None = None,
    allow_partial: bool = False,
) -> Dataset:
    data, nodes, affected = axis_context(x, dim)
    step = window_parameters(data, size, step, allow_partial)
    coordinates, unit, unit_latex, dtype, _ = numeric_coordinate(x, dim, nodes)
    rows, selections, starts, stops, numbers = window_plan(coordinates, size, step, allow_partial)
    ids = new_ids(len(rows))
    result = axis_dataset(data, nodes, affected, rows, selections, ids)
    provenance = window_metadata(data, rows, starts, stops, numbers, ids, unit, unit_latex, dtype)
    return result.assign(provenance)
