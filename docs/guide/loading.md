# Loading
Prepare data as a CSV table with a matching JSON schema. Array fields reference
numeric NPY or NPZ files.

```text
experiment/
├── samples.csv
├── samples.schema.json
└── arrays/
```

Omit `arrays/` for scalar tables. The [data format](../reference/import-format.md)
specifies both layouts and includes complete examples.

```python
import signalearn as sl

data = sl.load_data("experiment/samples.csv")
```

SignaLearn finds the schema beside the CSV. Input paths follow the working
directory; paths inside the manifest are relative to that manifest.

## Preparation
Define what a sample represents, its fields, units, coordinates, and any
existing partitions. Preserve the source values, array types, and axis order.
Unknown units or sampling rates should remain unspecified until verified.
A wavelength range alone does not establish exact band centres.

Decode source formats such as PNG, MP4, or IDX before import. SignaLearn reads
prepared numeric arrays, not those original formats. Cleaning, scaling,
resizing, imputation, and new partitions are separate analysis choices.
Keep original downloads and record preparation decisions in shared notes.

## Execution
SignaLearn uses Dask internally to defer loading and computation. Selecting
recordings usually leaves their signal arrays on disk. Accessing, plotting,
evaluating, or saving a result computes the values it needs; `.values` returns
NumPy arrays. No Dask-specific analysis code is required.

Some operations, including fitting a model, run immediately. Deferred loading
does not guarantee small batches or low memory use: this depends on the
operation and component. Keep source files unchanged while using a lazy dataset.
See [I/O](../reference/io.md#load_data) for validation and storage behaviour.
