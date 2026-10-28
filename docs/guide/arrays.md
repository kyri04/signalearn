# Arrays

Use `slice`, `crop`, and `window` to work inside each sample's arrays.
Selecting `data[:3]` instead selects three whole samples.

## Coordinates

This example creates two recordings with different lengths. Each signal's
`time` axis references its own time field:

```python
import numpy as np
import signalearn as sl

ids = ["a", "b"]
recordings = sl.Dataset({
    "signal": sl.Field(
        [np.ones(10), np.ones(15)], ids=ids, dims=("time",),
        coords={"time": {"field": "time"}}, unit="mV",
    ),
    "time": sl.Field(
        [np.arange(10) * 0.1, np.arange(15) * 0.1],
        ids=ids, dims=("time",), unit="s",
    ),
})
```

Matching axis names alone do not link fields. The explicit reference lets
operations update the signal and its time coordinates together.

## Slice and crop

Continuing with `recordings`:

```python
selected = sl.slice(recordings.signal, dim="time", indices=range(2, 6))
cropped = sl.crop(recordings.signal, dim="time", start=0.2, stop=0.5)
```

Slicing selects positions. Cropping selects coordinate values, including both
bounds. Here the bounds are seconds. Unrelated fields remain unchanged.

## Window

```python
windows = sl.window(recordings.signal, dim="time", size=0.4, step=0.2)
train, test = sl.split(windows, sizes=(0.5, 0.5), group=windows.source_id, seed=42)
```

Windows have new IDs and fields recording their source and boundaries. Group
related windows by recording or subject during evaluation to avoid leakage.

Window intervals include their start and exclude their end. Incomplete tails
are dropped unless `allow_partial=True`; no padding is added. Coordinates must
be regular and increasing, with size and step covering whole samples.
See the [array contracts](../reference/preprocess.md#slice) for restrictions.
