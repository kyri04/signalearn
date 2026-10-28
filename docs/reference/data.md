# Data

Public objects are defined in `signalearn.data`. They retain sample identity
and interpretation; their public `.values` interface returns NumPy values.

## Dataset

```python
sl.Dataset(fields, *, ids=None)
```

`fields` maps non-empty names to sequences, NumPy/Dask arrays, or existing
Fields. All have the same sample count. Existing Fields align by exact ID set;
raw sequences follow row order. Unspecified IDs come from the first Field, or
are newly generated when no Field supplies them. A non-nullable string `id`
column can supply IDs instead of the `ids` argument; supplying both is an error.
IDs are unique non-empty strings.

`data.ids` is the immutable ID tuple. `data.fields` is a read-only mapping
including `id`; `len(data)` counts samples. Names are preserved exactly. Brackets
support spaces, keywords, and names colliding with attributes or methods.

| Access | Return |
| --- | --- |
| `data.name`, `data["name"]` | Field in the current Dataset context. |
| `data[index]` | Sample at an integer position. |
| `data[slice_or_indices_or_mask]` | New Dataset view. |

Negative indices work. Integer sequences cannot repeat a sample. Eager boolean
masks must match the sample count; lazy masks are unsupported. Empty selections
retain field declarations. Views share storage and preserve ID order. Selection never restores excluded
samples or fields from an earlier dataset context.

### select

```python
data.select(*names)
```

Return only requested Fields, always retaining `id`. Samples are unchanged.
Hidden array coordinate dependencies remain attached.

### assign

```python
data.assign(fields=None, /, **updates)
```

Return a Dataset with added/replaced Fields. Existing Fields require the exact
sample ID set and are reordered automatically. Raw values require the same
length and follow row order. A positional mapping supports arbitrary names;
mapping and keyword updates cannot supply a name twice. `id` cannot be replaced.
The original Dataset remains unchanged.

### update

```python
data.update(fields=None, /, **updates)
```

Change display metadata on existing Fields. Each update maps attributes to
values. Accepted attributes are `unit`, `label`, `unit_latex`, `label_latex`,
`colour`, `colours`, `marker`, and `markers`. Category mappings merge; `None`
clears an attribute or entire mapping, while an empty mapping leaves entries
unchanged. Individual category entries cannot be `None`.

Unspecified attributes remain intact. Plain/formatted strings are independent;
clear an existing formatted label to use a replacement plain label. Updating a
unit does not convert values. Updates cannot add/rename fields, change values,
IDs, dtype, nullability, dimensions, coordinates, or prediction structure. No
values are computed; the returned Dataset shares storage. Input files and
previous plots are unchanged. Invalid metadata, unknown names, or duplicate
mapping/keyword updates raise errors.

## Field

```python
sl.Field(
    values, *, ids=None, dtype=None, unit=None, label=None,
    unit_latex=None, label_latex=None, colour=None, colours=None,
    marker=None, markers=None, nullable=None, dims=(), coords=None,
    shape=None, prediction=None,
)
```

`values` contains one scalar or array per sample. IDs are supplied or generated.
When constructed independently, a Field has no Dataset selection context; add
it to a Dataset before Field-driven whole-Dataset operations.

### Values

Scalars support logical `string`, `bool`, `int64`, and `float64` types. Types and
nullability can be inferred from in-memory values. All-missing scalar input
requires a dtype. Integers and floats may infer `float64`; strings, booleans,
and numbers cannot be mixed. Strings are never parsed as numbers. Scalar
missing values are `None`, require nullability, and preserve integer/boolean
identity through object storage. Numeric scalar values must be finite.

Arrays support `bool`, signed/unsigned 8/16/32/64-bit integers, and
`float16`/`float32`/`float64`. Supply non-empty ordered `dims`. NumPy/Dask batches
have a sample axis before these dimensions; sequences supply one rectangular
numeric array per sample. A declared dtype must match storage, not request a
cast. `shape` has one non-negative integer or `None` per dimension and validates
sizes without reshaping. Unknown Dask shapes are unsupported.

Floating arrays may declare `nullable=True` for NaN elements. Infinity, missing
whole arrays, masked arrays, complex/structured/string/object arrays, and nullable
integer/boolean arrays are unsupported. Empty array Fields need an explicit
dtype and enough structure to represent their shape.

### Coordinates

`coords` maps declared dimensions to inline numeric/string/boolean vectors or
references such as `{"field": "time"}`. Inline vectors are shared; mixed
booleans/numbers or strings/numbers are rejected. Referenced Fields are
one-dimensional numeric arrays with matching IDs and sample axis lengths.
References cannot be missing, ambiguous, cyclic, or self-referential. Numeric
coordinates must be finite even when measurement arrays permit missing values.

`field.dims` is a tuple. `field.coords` returns read-only inline vectors or
coordinate Fields; a sample view resolves the sample's coordinate value and
metadata. Dimension names alone do not establish alignment. Selecting only a
dependent Field retains hidden coordinate dependencies. Updating or replacing
an explicitly referenced coordinate in a Dataset updates its dependents.

### Access

Fields expose `ids`, `name`, `dtype`, `nullable`, `dims`, `coords`, `prediction`,
and the display attributes listed under `update`. A Field's name is established
by its Dataset. Unit/label attributes are strings or `None`; colours and markers
are Matplotlib strings. Category mappings use keys compatible with the value
dtype and may include unobserved categories. See [prediction metadata](predictions.md)
for the separate class display representation.

`field.values` and `field.to_numpy()` return independent arrays in current sample
order. Equal-shaped array samples stack; ragged samples form an object container
of independent numeric arrays. Integer Field indexing returns a raw scalar,
`None`, or array. Slices/index sequences/masks return Fields with metadata and
subset context. `len(field)` counts samples. Display, `list(field)`, NumPy
conversion, and integer indexing consume values; metadata access stays lazy.
NumPy `copy=False` is rejected. Raw array operations do not align IDs or infer units.
There is no persistent value cache; separate accesses may repeat computation.
Independent Fields support plots and transformations, but whole-dataset
selection requires a Dataset context.

NumPy construction copies input values. External Dask sources remain deferred
and must not change while in use. Object-backed scalar Dask input needs an
explicit logical dtype; validation occurs when consumed. Large Field displays
use bounded previews following NumPy print options, subject to upstream batch
semantics.

## Sample

`data[index]` returns a `Sample`. `sample.id` is the raw ID string;
`sample.fields` maps names to regular one-sample Fields. Named/bracket access
returns a value-like `SampleField`, including bracket access to `"id"`.
Both Dataset and Sample advertise field names through runtime completion.

## SampleField

Usually obtained through `data[index].name`. It retains identity and metadata
while displaying and supporting operations on the selected value.

`.value` returns the raw Python scalar/`None` or NumPy array. Scalar arithmetic,
comparison, formatting, rounding, and Python conversions produce ordinary
results without inferred units or IDs. The object itself is never `None`;
test `.value is None`. Scalar `.values` / `.to_numpy()` retain a sample axis
of length one, whereas `np.asarray(sample_field)` is zero-dimensional.

For an array SampleField, `.value`, `.values`, `.to_numpy()`, and NumPy conversion
omit the sample axis. `.shape` and `.ndim` describe that array; indexing its
contents and arithmetic return raw NumPy results. `len(sample_field)` still
counts its one sample. Use `sample.fields[name]` for the regular one-sample Field.

## Component outputs

A custom transformation can declare scalar and array outputs together. For a
spectrum with an established unit:

```python
def signalearn_outputs(self, inputs):
    unit = inputs["spectrum"]["unit"]
    return {
        "mean_absorbance": {"dtype": "float64", "unit": unit},
        "extrema": {
            "dtype": "float64", "dims": ("statistic",), "shape": (2,),
            "coords": {"statistic": ["minimum", "maximum"]}, "unit": unit,
        },
    }
```

This is an output declaration, not a standalone analysis. See
[Components](../extensions/components.md#outputs) for inheritance and shape rules.
