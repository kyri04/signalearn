# Data
A **Dataset** contains samples with named **Fields**. A sample might be a
participant, image, recording, or spectrum. Each field holds one scalar or
array per sample.

```python
import signalearn as sl

data = sl.Dataset({
    "temperature": sl.Field([36.8, 37.1], unit="degC"),
    "subject": ["S01", "S02"],
})
```

This dataset contains two samples. Each has a temperature and subject ID.
Pass fields directly to SignaLearn functions to retain their identity and metadata.

## Access
| Expression | Result |
| --- | --- |
| `data.temperature` | Field across samples. |
| `data[0]` | First sample. |
| `data[0].temperature` | One measurement, with metadata. |
| `data[0].temperature.unit` | `"degC"`. |
| `data.temperature.values` | NumPy array of values. |

`data[0].temperature.value` extracts the raw value. Integer Field indexing,
such as `data.temperature[0]`, also returns a raw value. Neither raw values nor
NumPy arithmetic align samples or infer units.

## Arrays
A sample can hold several fields, including arrays of different shapes.
Here each sample has a subject ID and a recording:

```python
recordings = sl.Dataset({
    "subject": ["S01", "S02"],
    "signal": sl.Field(
        [[0.2, 0.5, 0.1], [0.4, 0.3, 0.6, 0.2]],
        dims=("time",), unit="mV",
    ),
})
```

`dims` names the axes within a sample. The sample axis is separate. Coordinates
can describe positions along an axis, such as timestamps or wavelengths.
See [Arrays](arrays.md) for coordinate-based operations.

## Identity
Every sample has a unique string ID. SignaLearn aligns fields by these IDs,
including when rows arrive in a different order. Equal lengths alone do not
establish that observations correspond.

Supply IDs when importing related data separately. Otherwise, SignaLearn
generates them. Selection and predictions preserve IDs; new observations such
as windows get new IDs and retain their source IDs.

Data operations return new datasets. Use `assign` to add or replace fields and
`update` to change display metadata. The original dataset remains unchanged.

See the [data reference](../reference/data.md) for indexing, missing values,
assignment, and alignment rules.
