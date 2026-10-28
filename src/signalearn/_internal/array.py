from dask import compute
import numpy as np

from .data import materialise_values


def array_preview(field, display):
    if field.dims:
        from .tensor import tensor_preview

        return tensor_preview(field, display)
    options = np.get_printoptions()
    edge = options["edgeitems"]
    if len(field) <= max(options["threshold"], 2 * edge):
        return display(materialise_values(field))
    if edge:
        first, last = compute(field._values[:edge], field._values[-edge:])
        preview = np.concatenate((first, np.zeros(1, dtype=first.dtype), last))
        text = np.array2string(preview, separator=", ", threshold=2 * edge,
                               edgeitems=edge, prefix="array(" if display is repr else "")
    else:
        text = "[...]"
    if display is str:
        return text
    return f"array({text}, dtype={field._values.dtype.name}, shape=({len(field)},))"


class ArrayOperations:
    def __repr__(self):
        return array_preview(self, repr)

    def __str__(self):
        return array_preview(self, str)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("Converting a field to a NumPy array requires a copy")
        values = materialise_values(self)
        return values if dtype is None else values.astype(dtype, copy=False)

    def __iter__(self):
        values = materialise_values(self)
        return iter(values if self.dims else values.tolist())
