# I/O
Defined in `signalearn.io`. Figure saving is specified under
[save_plot](plotting.md#save_plot).

## load_data
```python
sl.load_data(path, *, schema=None)
```

Return a Dataset from a local CSV table/manifest and version-1 schema.

| Supplied path | Entry files |
| --- | --- |
| Dataset folder | Its `samples.csv` and `samples.schema.json`. |
| Its `arrays/` folder | The pair in its immediate parent. |
| CSV | Matching `.schema.json` beside it. |
| `.schema.json` | Matching CSV beside it. |

A folder named `arrays` uses its own standard entry files if either is present;
otherwise it uses its immediate parent. Missing required files raise
FileNotFoundError with their paths. No recursive search or alternate-pair
selection occurs.

`schema=` overrides the schema for a CSV or folder; supplying it alongside a
schema-file entry raises ValueError. Explicit paths follow the working directory
and expand `~`. Array paths remain relative to the resolved CSV. Custom basenames
work through explicit CSV/schema paths; folders require the standard filenames.
Only prepared `rows`/`arrays` layouts are supported. Raw files, arbitrary JSON
entries, schema dictionaries, inferred targets, and implicit cleaning are unsupported.

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
returns the estimator or `sl.Model` with fitted state and recorded
SignaLearn metadata, without fitting/applying it. Save leaves its supplied
instance unchanged. Path handling, existing-destination rejection, staging, and
failure cleanup follow dataset saving.

### Formats
Sklearn-compatible estimators support native composition and importable custom
estimators. Learned components must be fitted; stateless components need no
unnecessary fitting. Externally fitted estimators are allowed, with unknown
scientific metadata left unknown. `sl.Model` supports native PyTorch and Keras
artifacts. The custom `signalearn_mode` contract has no persistence integration.
Unsupported objects are rejected rather than arbitrarily pickled.

| Component | Native payload | Additional files |
| --- | --- | --- |
| Sklearn estimator | `model.joblib` | `model.json` |
| PyTorch `sl.Model` | `model.pt` state dictionary | `model.json`, `component.json` |
| Keras `sl.Model` | `model.keras` | `model.json`, `component.json` |

Share the entire directory. The version-1 `signalearn.model` manifest records
adapter, format, fixed payload name, component type, versions, and scientific
metadata. Only known adapters are selected; filesystem adapter imports are not
allowed. Sklearn uses Joblib with pickle protocol 5.

### Neural models
PyTorch requires an importable architecture `factory` and JSON-compatible
`factory_params` on `sl.Model`. Loading constructs the architecture and restores
its state dictionary using weights-only deserialization. The network object is
not pickled. Keras uses native `.keras` serialization and safe-mode loading;
custom objects must be available through Keras serialization.

`component.json` records input/output declarations and fitted input constraints.
Declarations must be JSON serializable. Loading restores an `sl.Model` with
inference weights on CPU. Artifacts do not embed preprocessing, custom source,
or trainers. Keras may store optimiser state in its native file, but SignaLearn
does not provide a training-resumption contract.

### Metadata
The manifest is authoritative. Input metadata records names, logical kinds, and
units in fitted order; target metadata records name, dtype, units, labels,
formatted attributes, colours, and markers. Unknown entries are null. Category
maps use typed category/display pairs, preserving booleans and large integers;
NumPy scalar keys become Python scalars.

Sklearn vector inputs additionally record their dimension, width, and optional
feature-coordinate values, kinds, and units in each input's `vector` entry.
These constraints are restored from the manifest and checked when applying the
model. Externally fitted models have no inferred scientific feature contract.

The native sklearn payload excludes `_signalearn_inputs` and `_signalearn_target`;
loading restores them from validated metadata. Native configuration, learned
state, and classes remain in the payload, potentially including retained
training observations. The artifact does not embed a notebook or preparation
code. Preprocessing and prediction components remain separate reusable objects.

### Compatibility
Record Python, SignaLearn, and the adapter's dependencies: sklearn/NumPy/SciPy/
Joblib, PyTorch/NumPy/Dask, or TensorFlow/Keras/NumPy/Dask. Sklearn custom top-level
classes also record their owning installed distribution when discoverable.
Require the same Python major/minor and exact recorded dependencies other than
SignaLearn. Python patch differences are permitted; the artifact format version
interprets SignaLearn metadata. Unknown versions, missing dependencies, invalid
metadata, or incompatible dependencies fail before native deserialization.
This is not an environment installer or complete custom-code lockfile.

Sklearn custom definitions and PyTorch architecture factories must resolve from
importable modules available when loading. Put them in the workspace extension
package. Source is not embedded; notebook definitions, lambdas, local definitions, and renamed/missing modules
are unsupported. Save validation checks importability, including referenced
custom callable parameters.

**Load artifacts only from trusted sources.** Joblib uses pickle machinery, and
architecture factories execute code. Manifest checks and native loading modes
do not make untrusted artifacts safe. See the
[native persistence guidance](https://scikit-learn.org/stable/model_persistence.html).

These artifacts support analysis/inference in compatible Python environments.
Device conversion and hardware optimisation belong to native deployment tools.
No general optimizer/scheduler/training-loop resumption contract is implied.
After cross-validation, explicitly fit final chosen components before saving.
