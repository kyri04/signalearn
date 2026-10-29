# Preprocessing
Defined in `signalearn.preprocess`. Data operations return new Datasets.
See [selection](../guide/selection.md) and [arrays](../guide/arrays.md)
for examples. `transform` is specified with [learning components](learning.md#transform).

## Selection context
Field-driven operations use the Field's current Dataset view, retaining visible
fields and selected IDs. Context follows Dataset/bracket/`fields` access,
slices, sample views, and bound coordinates. Independent Fields must first
belong to a Dataset. Context inspection is lazy; Field object identity is not
guaranteed across accesses. Use sample IDs for correspondence.

## filter
```python
sl.filter(x, value)
```

`x` is scalar. `value` is a compatible scalar or list/tuple/set/1D NumPy array
of accepted values. Matching uses equality/membership, retaining input order.
`None` matches missing values only when explicitly supplied. No match is a
valid empty Dataset; an empty request returns one without computing x.
Integers require integer matches; float Fields accept numeric matches;
strings/booleans are not coerced. Only x is computed.

## exclude
```python
sl.exclude(x, value)
```

Keep samples whose scalar `x` does not match `value`, using the same matching
rules as `filter`. Accept one value or a collection. Missing values remain unless
`None` is explicitly excluded. Preserve order, IDs, metadata, and current Dataset
context. An empty exclusion returns all selected samples without computing x;
excluding every value returns an empty Dataset. Only x is computed.

## between
```python
sl.between(x, lower=None, upper=None)
```

Select scalar numeric values within inclusive bounds in the existing unit.
Require at least one finite numeric bound and `lower <= upper`. Missing values
do not match. No conversion is inferred.

## where
```python
sl.where(x, condition)
```

Call a predicate once per selected sample, passing a Python scalar/`None` or
the sample's numeric array. Require one Python/NumPy boolean result. Numeric
truth values, arrays of booleans, and ambiguous results are errors. The predicate
handles missingness explicitly; exceptions propagate. Only x and its dependencies
are computed. Array content predicates read the selected arrays.

## sort
```python
sl.sort(x, descending=False)
```

Stable scalar numeric/string/boolean sorting, with missing values last in either
direction. Equal values retain order; strings use lexical order. Dates and
numeric strings are not interpreted. All context Fields follow the reordering;
only x is computed. Arrays require an explicit reduction first.

## take
```python
sl.take(x, ids)
```

`x` must be the Dataset's `id` Field. `ids` is one string, an eager sequence,
or a scalar string Field containing requested IDs. Return requested order.
Repeated or absent IDs are errors; empty requests return an empty Dataset.
Source observation values are unnecessary.

## sample
```python
sl.sample(data, n, seed=None)
```

Uniform sampling without replacement, returning observations in original order.
`n` is a non-negative integer no larger than the sample count. Zero is valid.
A non-negative integer seed is reproducible for the same order. Uses an
independent random generator without changing NumPy global state. No values
are computed; grouped/stratified sampling is not provided.

## split
```python
sl.split(data, sizes=(0.8, 0.2), *, group=None, seed=None, shuffle=True)
```

Return a tuple of Datasets, one per requested proportion, covering every sample
exactly once. Require at least two finite positive proportions summing to one.
Round counts down, then distribute remainders by largest fractional part,
breaking ties in proportion order. Empty resulting partitions are errors.
Within each partition, source row order is retained.

Shuffling randomises assignment with an optional non-negative integer seed.
Without shuffling, use existing order and omit seed. No target, timestamps, or
stratification are inferred. Ordinary splitting reads no values.

`group` is a scalar Field complete for the supplied Dataset's IDs. Extra IDs are
ignored and values align automatically.
Groups remain entirely within one partition. Proportions apply to distinct
group counts, so row proportions can differ. Without shuffling, groups follow
first appearance. Only group values are read.

## concat
```python
sl.concat(datasets)
```

Append a non-empty sequence of Datasets, preserving Dataset order and the first
field order. All inputs need identical field sets and disjoint IDs. Corresponding
fields require matching dtypes, ordered dimensions, units, and coordinate
declarations; per-sample shapes may vary. Referenced/hidden coordinate values
are concatenated with their samples. Different inline grids require explicit
preparation, rather than interpolation.

Nullability is combined. Category colours/markers merge, rejecting conflicts;
other explicit display attributes must agree, while unspecified ones may adopt
a supplied value. Prediction meaning must be compatible: per-class columns can
align by actual label to the first input's order, but missing classes or reversed
scalar score direction are errors. No ID regeneration, casting, padding,
normalisation, or resampling is inferred. Empty inputs within the sequence are
valid; storage is joined lazily.

## slice
```python
sl.slice(x, dim, indices)
```

Select inside each array sample along a named dimension. `indices` accepts an
integer, Python slice, range, integer sequence, or eager boolean mask. Negative
positions work. A single integer retains an axis of length one. Repeated or
out-of-bounds positions are errors, checked against every sample shape. Empty
axis selections retain samples with a zero-length axis. Shape checks are eager;
payloads stay lazy.

IDs, scalar fields, array dtypes, and valid metadata are preserved. Inline
coordinates follow the selection. Referenced coordinate Fields and all arrays
sharing that explicit coordinate follow together, including two axes referring
to one coordinate. Shared names or lengths alone do not couple arrays. Hidden
dependencies remain attached; ambiguous references require explicit resolution.
Selections compose, subject to upstream read/computation granularity.

## crop
```python
sl.crop(x, dim, start=None, stop=None)
```

Keep coordinate values within inclusive finite numeric bounds, in the declared
coordinate unit. Supply at least one bound with `start <= stop`. Coordinates
must be explicitly numeric inline vectors or referenced Fields; a coordinate
Field can itself be selected when referenced by another Field. Original order
is retained without sorting/interpolation. Samples may retain different lengths,
including zero. Coordinate values are read; other payloads stay lazy.
Shared-coordinate updates follow `slice`.

## window
```python
sl.window(x, dim, size, step=None, allow_partial=False)
```

Create new samples along an explicit numeric coordinate. `size` and `step` are
finite positive quantities in its unit; step defaults to size. Require strictly
increasing, regularly spaced coordinates and at least two points per non-empty
source. Size/step must be whole multiples of the sampling interval; no rounding,
padding, or resampling is inferred.

Windows start at the first coordinate and follow source order. Intervals are
half open; source coverage extends one sampling interval beyond the last point.
For points `0, ..., 9`, size 4 and step 2 produce `[0,4)`, `[2,6)`, `[4,8)`,
and `[6,10)`. Tail windows are excluded unless `allow_partial=True`, which
clips their stop to actual coverage. Short sources may emit none; empty sources
emit none. Coordinates retain their original values.

Every window has a new ID plus `source_id`, `window_index` (zero-based),
`window_start`, and `window_stop`. Boundaries retain units, using `int64` or
`float64`; integer overflow is rejected. Explicitly linked arrays are windowed
together; unrelated fields are copied from their source. Different inline grids
become internal per-sample coordinates. Payloads remain lazy while coordinates
are read to plan/validate windows.

Inputs already containing any provenance name are rejected, including nested
windowing. Later cropping/slicing preserves the original window boundaries as
provenance. Group by source recording or subject explicitly during evaluation.
Changing a prediction's `class` axis through any array operation is rejected.

## Folds and folds
```python
sl.Folds(data, indices)
sl.folds(data, splitter, *, target=None, group=None)
```

`Folds` is an immutable sequence of train/test Dataset pairs. Integer indexing
returns a pair; slicing returns a tuple of pairs. `.data` is the source Dataset;
`.indices` is the exact tuple of zero-based train/test positions. Each partition
is non-empty, unique, integer, in range, and disjoint from its counterpart.
Samples may be tested several times or never.

`folds` copies a native splitter exposing `split(X, y=None, groups=None)` and
supplies an empty matrix with the right sample count. Only explicit scalar,
target/group Fields complete for the supplied Dataset's IDs are computed and
aligned; extra IDs are ignored. Splitters can
use counts, order, labels, or groups. Feature-dependent selection needs a
dedicated integration or explicit `Folds` indices. A supplied group must remain
disjoint between train and test, even if the splitter ignores it. Native
memberships, gaps, and restricted training windows are preserved exactly.
Splitters are not fitted through `sl.fit`.
