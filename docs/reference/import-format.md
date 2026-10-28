# Data format

`load_data` accepts version-1 `rows` and `arrays` layouts. The same formats are
written by `save_data`. This reference specifies the on-disk contract; start
with [preparation](../guide/loading.md) for the workflow. Numeric
arrays represent images, recordings, spectra, or other explicitly structured
measurements. The old `recordings` layout name is unsupported.

## Requirements

- The sample boundary is explicit: one table row for `rows`, or one manifest
  row for `arrays`. One image, utterance, spectrum, or recording can be a sample.
  Import does not split recordings into windows or videos into frames.
- Tables are UTF-8 CSV files with a comma delimiter and one non-empty, unique
  column name per column. Standard CSV quoting handles commas and quotes inside
  values. These rules apply to scalar tables and array manifests. Array values
  use the explicitly supported binary formats described below.
- Headers are field or source-column names. Units, dimensions, and display labels
  are supplied as metadata, rather than inferred from header text.
- A companion JSON schema supplies the interpretation needed for the layout.
  Prepared datasets use `samples.csv` and `samples.schema.json` as their entry
  files for both layouts. Array datasets additionally use an `arrays/` folder
  for their prepared array files.
- Field names are non-empty, unique strings. Names are preserved exactly;
  `data["field name"]` supports names unsuitable for attribute access. `id` is
  reserved for sample identity. Array storage columns have explicit roles in
  source declarations; names such as `file` have no automatic interpretation.
- Targets and grouping fields are optional, ordinary fields. Names such as
  `label` and `subject` do not automatically designate learning targets or split
  groups; the calling operation makes those choices explicit.

Both layouts require a schema, including a simple scalar table.
`sl.load_data("samples.csv")` reads `samples.schema.json` alongside it.
Use `sl.load_data("samples.csv", schema="other.schema.json")` to select a
different schema file. Supplying a schema dictionary directly is not supported.
The prepared filename convention does not restrict the importer: an explicitly
supplied CSV with another name still uses its corresponding `.schema.json` file.

The scalar CSV importer reads and validates the table eagerly. Returned fields
use Dask internally; subsequent selection remains lazy. Accessing a field's
`.values` materialises its selected samples as an independent NumPy array.
It does not yet provide disk-backed or chunked CSV reading. Missing values are
represented as `None`; nullable fields use object storage to preserve integer
and boolean values without substituting a floating-point missing marker.

## Schema

`version` is `1`. `layout` is `rows` or `arrays`. A field
declaration contains a `dtype` and may include display metadata and `nullable`:

| Attribute | Meaning |
| --- | --- |
| `dtype` | Required value type. |
| `unit`, `label` | Units and display label. |
| `unit_latex`, `label_latex` | Formatted display strings. |
| `colour`, `marker` | Series appearance. |
| `colours`, `markers` | Category mappings. |
| `nullable` | Missing values allowed. |
| `missing_values` | Additional missing CSV tokens. |
| `encoding` | JSON string encoding. |
| `prediction` | Prediction meaning. |

Scalar CSV types are `string`, `bool`, `int64`, and `float64`; array types are
listed under [Shape](#shape). All attributes except `dtype` are optional.
`nullable` defaults to `false`; recognised missing values require it to be `true`.
`missing_values` is a list of literal tokens. `encoding="json"` is available
only for scalar strings, preserving empty strings separately from missing cells.
Prediction metadata describes probabilities, decision scores, or logits.

Units, labels, and formatted attributes are strings. Omitted or null units are
unspecified; labels fall back to field names. Plots prefer formatted attributes.
Colours and markers use Matplotlib strings; category mappings are JSON objects.
Metadata declarations never convert values.

Formatted display strings use Matplotlib Mathtext syntax, with dollar signs
around mathematical expressions. JSON backslashes must be escaped. For example:

```json
{
  "dtype": "float64",
  "label": "Power spectral density",
  "label_latex": "$S_{xx}(f)$",
  "unit": "uV^2/Hz",
  "unit_latex": "$\\mu\\mathrm{V}^{2}/\\mathrm{Hz}$",
  "colour": "#0072B2"
}
```

A string category field can declare `"colours": {"Adelie": "#0072B2",
"Gentoo": "#009E73"}`. JSON object keys are strings: for numeric and boolean
fields, keys are parsed using the declared dtype, for example `"1"` for an
integer and `"true"` for a boolean. Duplicate categories after parsing are
rejected. Missing values do not have category entries. Mappings may include
categories absent from the current table, and are retained when selecting a
subset. Invalid colours or incompatible category keys are schema errors.

Marker metadata follows the same convention. For example,
`"marker": "o"` selects circles for one series, and
`"markers": {"Adelie": "o", "Chinstrap": "s", "Gentoo": "^"}` maps categories
to circles, squares, and triangles. Marker strings must be valid Matplotlib
markers. Category keys are parsed using the field dtype and duplicate converted
keys are rejected. Selection preserves the mapping, including absent categories.
Size, opacity, fonts, and grids are configured in the shared plotting style
rather than individual field declarations.

Display metadata does not change imported values or units and does not assign
a target or grouping role. Grouping is still chosen by the plotting call.

Scalar CSV string values retain their text, including leading zeros. Boolean values are
written as `true` or `false`. Numeric columns must contain values compatible
with their declared type, with no thousands separators or units appended to
numbers. Undeclared type conversion is an error.

An empty CSV cell denotes a missing value. To distinguish an actual empty string
from missing text, string declarations may set `"encoding": "json"`. In that
case every non-missing cell must be a JSON string literal: `""` represents an
empty string, `"hello"` represents the text hello, and an empty CSV cell remains
missing. Standard CSV quoting still applies around the cell. JSON numbers,
booleans, arrays, objects, and `null` are not strings and are rejected. The
encoding option is supported only for scalar strings. Without it, strings are
literal text as before.

Other tokens such as `NA`, `?`, or `-999`
are not automatically treated as missing. The schema may declare a field's
`missing_values` list to recognise additional literal tokens. Missing values
require `nullable: true` and remain missing after import; rows are not dropped
or values imputed. Non-finite numeric tokens must be explicitly designated as
missing or are rejected by the importer.

Every CSV column must be accounted for: by `fields` in `rows`, or by
`sample_fields` or array `source` declarations in `arrays`. Missing, duplicated,
or undeclared columns are errors, including an accidental spreadsheet index.
Column order does not matter when columns are selected by name. Unknown schema
keys or unsupported schema versions are errors.

## Tables

Each row represents one observation and each declared field contains one scalar
value for that observation. For example, `samples.csv`:

```text
id,subject,temperature,amplitude,label
sample_001,S01,36.8,0.12,rest
sample_002,S02,37.1,0.18,active
```

Its companion `samples.schema.json`:

```json
{
  "version": 1,
  "layout": "rows",
  "fields": {
    "id": {"dtype": "string"},
    "subject": {"dtype": "string"},
    "temperature": {"dtype": "float64", "unit": "degC"},
    "amplitude": {"dtype": "float64", "unit": "mV"},
    "label": {"dtype": "string"}
  }
}
```

The `fields` keys match the column names. Scalar fields have no dimensions within
a sample, so their `dims` are empty and no coordinates are required. Each field
contains two sample values in this example.

Arrays, Python expressions, and JSON containers embedded in CSV cells are outside
this layout. They must not be evaluated or silently treated as signal arrays.
JSON string literals are supported only through the explicit string encoding
described above.

## Arrays

The manifest supplies scalar fields and literal references to numeric arrays.
The schema defines how each reference becomes a field value. File order,
directory contents, filenames, and matching array lengths do not establish
sample identity or interpretation.

Version-1 array schemas require `version`, `layout`, `sample_fields`,
and `fields`, and optionally accept `visible_fields`. `sample_fields` declares manifest columns using the scalar CSV
conventions above; it may be empty. `fields` is a non-empty mapping of array
field names to declarations. Names cannot collide between these two groups,
and array fields cannot be named `id`.

When supplied, `visible_fields` is an ordered list of unique declared field names,
including `id` and all scalar sample fields. Unlisted array fields must be
coordinate dependencies reachable from visible array fields. They are loaded
and bound as coordinates but omitted from the dataset's visible fields. This
lets saved subsets preserve their coordinate dependencies and visible field
order. If omitted, all declared fields remain visible as before.

### Storage

Supported storage formats are `.npy` for one numeric array and `.npz` for
named numeric arrays. Original formats such as IDX, JPEG, video containers,
or dataset-specific text need an explicit preparation step or a future
documented reader. A file extension alone does not establish its meaning.

Each array field requires a `source` object:

| Attribute | Meaning |
| --------- | ------- |
| `format` | Required: `npy` or `npz`. |
| `path_column` | Required file-path column. |
| `key` | Array name within an NPZ archive. |
| `index_column` | Optional leading-axis index column. |

`key` is required for NPZ and prohibited for NPY. It must identify the exact
array name in the archive. No other source attributes are allowed. Referenced column names and archive
keys must be non-empty strings.

Relative paths resolve against the manifest's directory; absolute paths are
literal. Every referenced file must exist. Preparation should use relative paths
so a prepared dataset can be moved together. Paths are not glob patterns,
expressions, URLs, or instructions to search other folders.

Without `index_column`, the entire selected array is one sample's field value.
With it, the importer selects `stored_array[index]`: that leading storage axis
is removed before validating the field's dimensions and shape. Indices are
non-negative decimal integers, zero-based, non-missing, and in bounds. This is
an explicit source position, not a sample ID or a dimension within the sample.

Several samples can share one file, and several fields can reference different
keys in the same archive. Source columns are loading instructions and are not
automatically materialised as ordinary fields. A column can also be declared
in `sample_fields` when it is deliberately needed as an analysis field. All
manifest columns must be declared or referenced; unused columns are errors.
Unreferenced archive keys are not imported as fields or assigned meaning.

Arrays must load without Python object deserialisation (`allow_pickle=False`).
Object arrays, structured arrays, complex arrays, string arrays, and pickled
objects are outside this array contract. A variable-length dataset
uses separate rectangular arrays for its samples rather than an object array
or automatic padding inside one file.

### Shape

Each array declaration requires `dtype` and `dims` in addition to `source`.
Array dtypes are `bool`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`,
`uint32`, `uint64`, `float16`, `float32`, or `float64`. The declaration must
match the stored logical dtype; import does not cast pixels or measurements
to a common dtype. These additional dtypes apply to array fields, not scalar
CSV fields. Byte order is a storage detail, not a different scientific dtype.

`dims` is a non-empty list of unique, non-empty dimension names, in the exact
order of the axes within one sample. Its length must match the selected array's
rank. Names are descriptive, not special data-type switches:

| Example | Dimensions within one sample |
| ------- | ---------------------------- |
| Multichannel signal | `["channel", "time"]` |
| Spectrum | `["band"]` or `["wavelength"]` |
| Grayscale image | `["height", "width"]` |
| Colour image | `["height", "width", "channel"]` |
| Video | `["time", "height", "width", "channel"]` |

An optional `shape` list, with the same length as `dims`, asserts expected
axis lengths in dimension order.
Each entry is a non-negative integer or `null` for an unconstrained length.
For example, `[28, 28]` fixes an image size and `[12, null]` allows utterances
of different frame counts. If omitted, axis lengths may vary across samples.
Shape is a validation constraint, not an instruction to reshape, resize, or
flatten values. Axis order must already match the declared dimension order.

Display metadata follows the common field conventions. Each field has at most
one declared value unit. Quantities with different units, such as force and
torque, should be separate fields unless a future explicit representation
supports their differing units.

### Coordinates

`coords` is optional. It maps a subset of the declared dimension names to either
a one-dimensional JSON list of finite numbers, strings, or booleans, or a
reference of the form `{"field": "time"}`. An empty vector is valid for a
zero-length axis. Inline coordinates must have one
homogeneous scalar kind, allowing integer and floating-point numbers together.
Boolean coordinates cannot be mixed with numeric values.
A referenced coordinate must be another declared one-dimensional numeric array
field in the same sample. Its own field metadata supplies any coordinate unit.
References cannot be self-references or form cycles.

Coordinate lengths must match their corresponding axes after source selection.
Coordinates are resolved for each sample, so recordings can have different
timestamps or lengths. A shared dimension name alone does not imply that two
fields share coordinates or are aligned. Coordinate references explicitly
establish that relationship; no interpolation or unit conversion is inferred.

Coordinates must not be missing. The importer does not infer sampling rates,
time origins, wavelength grids, pixel spacing, or colour-channel meanings.
When physical time or wavelength coordinates are not established by the source,
omit them and retain explicitly named axes or band/frame indices. Operations
requiring increasing time, regular sampling, or calibrated coordinates validate
those requirements separately. A coordinate declaration never resamples data.

Array fields default to `nullable: false`. For floating-point arrays,
`nullable: true` permits NaN elements as missing measurements; infinity is
always invalid. The storage retains its declared floating dtype. These NaNs
must be identified as missing by operations, rather than treated as finite
measurements. Version 1 does not support missing whole arrays or missing
integer/boolean elements; these require an explicit future mask representation.
`missing_values` is a CSV-only option and is not valid on array declarations.

Array declarations allow exactly `source`, `dtype`, `dims`, optional `shape`,
`coords`, `nullable`, the common display attributes, and optional `prediction` metadata. Unknown attributes are
errors. Missing paths or indices are errors even if an array is nullable.


## Example

For MNIST, samples share two image arrays. The file layout is:

```text
mnist/
├── samples.csv
├── samples.schema.json
└── arrays/
    ├── train_images.npy
    └── test_images.npy
```

Example manifest rows:

```text
id,image_file,image_index,digit,partition
mnist_train_00000,arrays/train_images.npy,0,5,train
mnist_train_00001,arrays/train_images.npy,1,0,train
mnist_test_00000,arrays/test_images.npy,0,7,test
```

The matching schema is:

```json
{
  "version": 1,
  "layout": "arrays",
  "sample_fields": {
    "id": {"dtype": "string"},
    "digit": {"dtype": "int64", "label": "Digit"},
    "partition": {"dtype": "string", "label": "Source partition"}
  },
  "fields": {
    "image": {
      "source": {
        "format": "npy",
        "path_column": "image_file",
        "index_column": "image_index"
      },
      "dtype": "uint8",
      "dims": ["height", "width"],
      "shape": [28, 28],
      "label": "Handwritten digit"
    }
  }
}
```

The original batch shapes are `(60000, 28, 28)` and `(10000, 28, 28)`.
Selecting each manifest index removes that leading storage axis, leaving one
`(28, 28)` image. Intensities retain source bytes from 0 to 255. No new channel
axis, unit, scaling, resize, or partition assignment is inferred. IDs include
the partition because source indices restart in each file.

## Validation

If an `id` column is supplied, declare it as a non-nullable string field. IDs
must be non-empty and unique across the dataset. Preserve supplied IDs exactly.
If the column and declaration are both omitted, SignaLearn generates IDs at
construction. Independently importing the same files is not guaranteed to
regenerate identical IDs, so supply IDs when matching across imports matters.

Import must reject malformed CSV, inconsistent row widths, duplicate IDs,
incompatible types, broken paths, absent archive keys, invalid source indices,
invalid coordinate references, and dtype, rank, or shape mismatches. Errors
should identify the source file, manifest row or sample ID, field, and expected
structure where applicable. Schema, path, index-format, and available array
header checks should run before consuming values. Validation requiring array
contents, archive decompression, or coordinates may occur when those values
are consumed.

Array imports parse the manifest and inspect referenced array headers, caching
each file/key header during loading. Paths, source indices, dtype, rank, shape,
and coordinate lengths are checked without materialising numeric payloads.
Values remain on disk until consumed. `.npy` sources are memory-mapped read-only
when needed; selecting one sample does not copy the full file. An `.npz` member
is decompressed as a whole when needed. Fields sharing that member reuse its
read within a joint computation.

Dask executes the selected sample tasks. Each consumed value is validated and
copied before being returned to the user. Referenced coordinate values are
checked alongside the samples that require them. Metadata access and sample
selection do not consume array values. A successful load does not establish
that every deferred value has already been validated. Source files must remain
unchanged while a lazy dataset is in use.

Import never silently repairs, drops, or reinterprets invalid observations.
Schema validation and mechanical conversion are separate from any subsequent
preprocessing chosen by the user.


## Predictions

The optional `prediction` entry identifies probabilities, decision scores,
or logits independently of value dtype. Its complete schema and typed class
display mappings are specified under [prediction metadata](predictions.md#schema).
See [preparation](../guide/loading.md) for other data types.
