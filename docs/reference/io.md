# I/O
Defined in `signalearn.io`. Figure saving is specified under
[save_plot](plotting.md#save_plot).

## load_data
```python
sl.load_data(path, *, schema=None)
```

Return a Dataset from a local CSV table/manifest and version-1 schema. With no
schema argument, replace the CSV suffix with `.schema.json` beside the CSV.
Both explicit paths follow the process's working directory. Only local CSV
and documented `rows`/`arrays` layouts are supported; there is no directory
search, schema-dictionary input, inferred target, or implicit data cleaning.

CSV cells are parsed/validated eagerly. Array references and available headers
are checked eagerly, while numeric payloads remain deferred. NPY reads use
read-only memory maps; NPZ reads decompress the requested member. Shared reads
can be reused within a joint computation. See [import format](import-format.md).

Invalid schema, types, IDs/headers, columns, row widths, malformed CSV, source
indices, or array structure raise contextual errors. Missing CSV/schema files
raise FileNotFoundError. Deferred value/coordinate checks run when consumed;
successful loading does not validate every payload. Sources must remain unchanged.

## save_data
```python
sl.save_data(data, path)
```

`data` is a Dataset; `path` is a new destination directory. Return the absolute
Path to its `samples.csv`. Relative paths follow the working directory, `~`
expands, and missing parents are created. Existing files/directories/symlinks
raise FileExistsError. A complete staged sibling directory is published after
writing succeeds; failures remove staged files.

Write `samples.csv`, `samples.schema.json`, and `arrays/` when needed. Preserve
selected order, IDs, field order, scalar/array dtypes, missing values, dimensions,
coordinates, display metadata, provenance, and prediction meaning. Hidden
coordinate dependencies remain bound and are recorded through `visible_fields`;
conflicting hidden references are rejected. No per-dataset README/checksum is
required. Keep preparation notes in shared documentation rather than generating
per-dataset README or checksum files. Model/Python-object serialization is separate.

Saving eagerly materialises the selection in batches of up to 1,024 samples
and approximately 8 MiB of array values. A larger sample is processed alone.
Native readers and upstream batch components may require larger reads or
computations. String Fields are additionally checked for empty strings.
Equal-shaped samples share indexed NPY payloads; ragged shapes use separate
numeric files without padding/object serialization. Empty datasets retain
available declarations/shape hints and require no payloads.

The folder is self-contained and uses relative array paths. Move/share all its
files. Scalar strings containing paths remain literal; referenced files are not
implicitly copied. Empty strings use explicit JSON string encoding to remain
distinct from missing CSV cells. Typed category keys survive round trips.
Values are saved without reshaping, scaling, casting, or resampling.

## Models
```python
sl.save_model(model, path)
sl.load_model(path)
```

Save returns the absolute new directory Path. Load accepts that directory and
returns the original native component type with fitted state and recorded
SignaLearn metadata, without fitting/applying it. Save leaves its supplied
instance unchanged. Path handling, existing-destination rejection, staging, and
failure cleanup follow dataset saving.

### Support
The current persistence adapter supports sklearn-compatible components exposing
get_params/fit, including supported native composition and importable custom
estimators. Learned components must be fitted; native stateless components can
be saved without unnecessary fitting. Externally fitted components are allowed;
unknown scientific metadata remains unknown. The custom `signalearn_mode`
transform contract and native PyTorch/Keras persistence are not implemented.
Unsupported objects are rejected rather than arbitrarily pickled.

```text
classifier/
├── model.json
└── model.joblib
```

The native payload is Joblib using pickle protocol 5. The manifest is version 1,
kind `signalearn.model`, adapter `sklearn`, format `joblib`, with a fixed payload
name, component module/qualified name, dependency versions, and metadata.
Only known adapters are selected; arbitrary filesystem adapter imports are not
allowed. Share the entire directory.

### Metadata
The manifest is authoritative. Input metadata records names, logical kinds, and
units in fitted order; target metadata records name, dtype, units, labels,
formatted attributes, colours, and markers. Unknown entries are null. Category
maps use typed category/display pairs, preserving booleans and large integers;
NumPy scalar keys become Python scalars.

The native sklearn payload excludes `_signalearn_inputs` and `_signalearn_target`;
loading restores them from validated metadata. Native configuration, learned
state, and classes remain in the payload, potentially including retained
training observations. The artifact does not embed a notebook or preparation
code. Preprocessing and prediction components remain separate reusable objects.

### Compatibility
Record Python, SignaLearn, sklearn, NumPy, SciPy, and Joblib versions, plus an
installed distribution owning a non-sklearn top-level class when discoverable.
Require the same Python major/minor and exact recorded dependencies other than
SignaLearn. Python patch differences are permitted; the artifact format version
interprets SignaLearn metadata. Unknown versions, missing dependencies, invalid
metadata, or incompatible dependencies fail before native deserialization.
This is not an environment installer or complete custom-code lockfile.

Custom classes/functions must resolve from importable modules available when
loading. Put them in the workspace extension package. Source is not embedded;
notebook definitions, lambdas, local definitions, and renamed/missing modules
are unsupported. Save validation checks importability, including referenced
custom callable parameters.

**Load artifacts only from trusted sources.** Joblib uses pickle machinery;
manifest and version checks do not make an untrusted payload safe. See the
[native persistence guidance](https://scikit-learn.org/stable/model_persistence.html).

These artifacts support analysis/inference in compatible Python environments.
Device conversion and hardware optimisation belong to native deployment tools.
No general optimizer/scheduler/training-loop resumption contract is implied.
After cross-validation, explicitly fit final chosen components before saving.
