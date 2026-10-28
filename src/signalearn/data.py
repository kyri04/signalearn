from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, overload

import dask.array as da
import numpy as np

from ._internal.array import ArrayOperations
from ._internal.data import (
    align_field, bound_field, build_fields, clone_field, dataset_fields, dataset_view, field_attributes, lookup_field,
    initialise_sample_field, materialise_value, materialise_values, new_ids,
    positions, prepare_values, take_field, update_dataset_metadata, validate_ids, validate_names,
)
from ._internal.scalar import ScalarOperations
from ._internal.metadata import prepare_colours, prepare_markers, validate_metadata
from ._internal.prediction import prepare_prediction
from ._internal.tensor import (
    ArrayStorage, check_coordinates, coordinate_specs, dimensions, field_coordinates, prepare_arrays,
)

__all__ = ["Dataset", "Field", "Sample", "SampleField"]


@dataclass(frozen=True, init=False, eq=False, repr=False)
class Field(ArrayOperations):
    _values: da.Array | ArrayStorage
    _dims: tuple[str, ...]
    _shape: tuple[int | None, ...] | None
    _coords: Mapping[str, Any]
    _coordinate_fields: Mapping[str, "Field"]
    _dataset: "Dataset | None"
    ids: tuple[str, ...]
    dtype: str
    unit: str | None
    label: str | None
    unit_latex: str | None
    label_latex: str | None
    colour: str | None
    colours: Mapping[str | bool | int | float, str]
    marker: str | None
    markers: Mapping[str | bool | int | float, str]
    nullable: bool
    name: str | None
    prediction: Mapping[str, Any] | None

    def __init__(
        self, values: Sequence[Any] | np.ndarray | da.Array, *,
        ids: Sequence[str] | None = None, dtype: str | None = None,
        unit: str | None = None, label: str | None = None,
        unit_latex: str | None = None, label_latex: str | None = None,
        colour: str | None = None,
        colours: Mapping[str | bool | int | float, str] | None = None,
        marker: str | None = None,
        markers: Mapping[str | bool | int | float, str] | None = None,
        nullable: bool | None = None,
        dims: Sequence[str] = (), coords: Mapping[str, Any] | None = None,
        shape: Sequence[int | None] | None = None,
        prediction: Mapping[str, Any] | None = None,
    ) -> None:
        dims, shape = dimensions(dims, shape)
        validate_metadata(dtype, unit, label, nullable, unit_latex, label_latex, colour, marker, array=bool(dims))
        storage, dtype, nullable = (prepare_arrays(values, dtype, nullable, dims, shape) if dims
                                    else prepare_values(values, dtype, nullable))
        length = int(storage.shape[0])
        identity = new_ids(length) if ids is None else validate_ids(ids, length)
        for key, value in {
            "_values": storage, "ids": identity, "dtype": dtype, "unit": unit,
            "label": label, "nullable": nullable, "name": None,
            "unit_latex": unit_latex, "label_latex": label_latex,
            "colour": colour, "colours": prepare_colours(colours, dtype),
            "marker": marker, "markers": prepare_markers(markers, dtype),
            "_dims": dims, "_shape": shape, "_coords": coordinate_specs(coords, dims),
            "_coordinate_fields": MappingProxyType({}),
            "_dataset": None,
            "prediction": prepare_prediction(prediction, dims, coords, dtype=dtype, shape=shape, nullable=nullable),
        }.items():
            object.__setattr__(self, key, value)
        if dims:
            check_coordinates(self)

    @property
    def values(self) -> np.ndarray:
        return materialise_values(self)

    @property
    def dims(self) -> tuple[str, ...]:
        return self._dims

    @property
    def coords(self) -> Mapping[str, Any]:
        return field_coordinates(self)

    def to_numpy(self) -> np.ndarray:
        return self.values

    def __len__(self) -> int:
        return len(self.ids)

    @overload
    def __getitem__(self, selector: int | np.integer) -> str | bool | int | float | np.ndarray | None: ...

    @overload
    def __getitem__(self, selector: slice | Sequence[int] | Sequence[bool] | np.ndarray) -> "Field": ...

    def __getitem__(self, selector: Any) -> "Field | str | bool | int | float | np.ndarray | None":
        indices, scalar = positions(selector, len(self))
        if scalar:
            return materialise_value(self, indices[0])
        return take_field(self, indices)


@dataclass(frozen=True, init=False, eq=False, repr=False)
class SampleField(ScalarOperations, Field):
    def __init__(self, field: Field, index: int = 0) -> None:
        initialise_sample_field(self, field, index)

    @property
    def value(self) -> str | bool | int | float | np.ndarray | None:
        return materialise_value(self, 0)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._values.items[0].shape) if self.dims else ()

    @property
    def ndim(self) -> int:
        return len(self.dims)

    @property
    def values(self) -> np.ndarray:
        return self.value if self.dims else materialise_values(self)

    def __getitem__(self, selector: Any) -> Any:
        return self.value[selector] if self.dims else super().__getitem__(selector)

    def __iter__(self):
        return iter(self.value) if self.dims else super().__iter__()


@dataclass(frozen=True, init=False, eq=False)
class Dataset:
    _fields: Mapping[str, Field]
    _ids: tuple[str, ...]

    def __init__(
        self, fields: Mapping[str, Sequence[Any] | np.ndarray | da.Array | Field], *,
        ids: Sequence[str] | None = None,
    ) -> None:
        fields, identity = build_fields(fields, ids, Field)
        object.__setattr__(self, "_fields", fields)
        object.__setattr__(self, "_ids", identity)

    @property
    def fields(self) -> Mapping[str, Field]:
        return dataset_fields(self)

    @property
    def ids(self) -> tuple[str, ...]:
        return self._ids

    def select(self, *names: str) -> "Dataset":
        selected = {"id": self._fields["id"]}
        for name in names:
            selected[name] = self._fields[name]
        return dataset_view(self, selected, self.ids)

    def assign(self, fields: Mapping[str, Any] | None = None, /, **updates: Any) -> "Dataset":
        additions = {} if fields is None else dict(fields)
        if additions.keys() & updates.keys():
            raise ValueError("A field was supplied both in the mapping and as a keyword")
        additions.update(updates)
        validate_names(additions)
        if "id" in additions:
            raise ValueError("Assignment cannot replace sample IDs")
        result = dict(self.fields)
        for name, value in additions.items():
            field = align_field(value, self.ids) if isinstance(value, Field) else Field(value, ids=self.ids)
            result[name] = clone_field(field, name=name)
        return dataset_view(self, result, self.ids)

    def update(
        self, fields: Mapping[str, Mapping[str, Any]] | None = None, /,
        **updates: Mapping[str, Any],
    ) -> "Dataset":
        return update_dataset_metadata(self, fields, updates)

    def __len__(self) -> int:
        return len(self.ids)

    @overload
    def __getitem__(self, selector: str) -> Field: ...

    @overload
    def __getitem__(self, selector: int | np.integer) -> "Sample": ...

    @overload
    def __getitem__(self, selector: slice | Sequence[int] | Sequence[bool] | np.ndarray) -> "Dataset": ...

    def __getitem__(self, selector: Any) -> "Field | Sample | Dataset":
        if isinstance(selector, str):
            return bound_field(self._fields[selector], self)
        indices, scalar = positions(selector, len(self))
        if scalar:
            return Sample(self, indices[0])
        fields = {name: take_field(field, indices) for name, field in self.fields.items()}
        return dataset_view(self, fields, tuple(self.ids[i] for i in indices))

    def __getattr__(self, name: str) -> Field:
        return bound_field(lookup_field(object.__getattribute__(self, "_fields"), name), self)

    def __dir__(self) -> list[str]:
        return field_attributes(self, self._fields)

    def _ipython_key_completions_(self) -> list[str]:
        return list(self._fields)

    def __repr__(self) -> str:
        return f"Dataset(samples={len(self)}, fields={list(self.fields)!r})"


@dataclass(frozen=True, eq=False)
class Sample:
    _dataset: Dataset
    _index: int

    def __post_init__(self) -> None:
        indices, scalar = positions(self._index, len(self._dataset))
        if not scalar:
            raise TypeError("A sample requires one integer position")
        object.__setattr__(self, "_index", indices[0])

    @property
    def id(self) -> str:
        return self._dataset.ids[self._index]

    @property
    def fields(self) -> Mapping[str, Field]:
        return MappingProxyType({name: take_field(field, [self._index]) for name, field in self._dataset.fields.items()})

    def __getitem__(self, name: str) -> SampleField:
        return SampleField(self._dataset[name], self._index)

    def __getattr__(self, name: str) -> SampleField:
        return SampleField(bound_field(lookup_field(self._dataset._fields, name), self._dataset), self._index)

    def __dir__(self) -> list[str]:
        return field_attributes(self, self._dataset.fields)

    def _ipython_key_completions_(self) -> list[str]:
        return list(self._dataset.fields)

    def __repr__(self) -> str:
        return f"Sample(id={self.id!r}, fields={list(self._dataset.fields)!r})"
