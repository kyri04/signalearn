# Components

Use `sl.transform` with a component that declares its inputs and outputs.
The same public contract applies to human-written and agent-written code.

## Layout

```text
sleep_study/
├── analysis.py
└── signalearn_sleep_study/
    ├── __init__.py
    ├── preprocess.py
    ├── learning.py
    ├── plotting.py
    └── _internal/
        ├── __init__.py
        └── adapters/
            └── __init__.py
```

Create only modules you need. Put public reusable operations/components in the
relevant module, and every helper in the local `_internal/` package. Keep the
package importable from your analysis environment. Workspace extensions do not
modify installed SignaLearn code.

## Example

Place this component in your project's `preprocess.py` module:

```python
class RemoveOffset:
    signalearn_mode = "sample"
    signalearn_inputs = {"signal": {"dims": ("time",), "dtype": "float64", "unit": "mV"}}

    def __init__(self, offset):
        self.offset = offset

    def signalearn_outputs(self, inputs):
        return {"signal": {"source": "signal"}}

    def transform(self, inputs):
        return {"signal": inputs["signal"] - self.offset}
```

Using the `recordings` dataset from [Arrays](../guide/arrays.md#coordinates):

```python
processed = sl.transform(recordings.signal, RemoveOffset(offset=0.1))
updated = recordings.assign(signal=processed.signal)
```

This stateless transformation preserves units, shape, and sample IDs. Learned
components must be fitted separately on the training partition.

## Interface

| Member | Purpose |
| --- | --- |
| `signalearn_mode` | `"sample"` or `"batch"`. |
| `signalearn_inputs` | Input declarations. |
| `signalearn_outputs(inputs)` | Output declarations. |
| `transform(inputs, **parameters)` | Output values. |

Each input requires `dims`; `dtype`, `unit`, and `shape` are optional exact
constraints. Outputs must be a non-empty mapping declared from metadata before
computation. `transform` returns exactly those names and one value per sample.

Input names must match the selected Fields. IDs align automatically, and order
follows the declaration. Dimensions match exactly, including order; scalars use
`()`. No casting or unit conversion is inferred.

`signalearn_outputs` receives dictionaries containing dtype, dims, nullable,
coords, display attributes, common per-sample shape (or None), and exact shapes
for each sample. Scalars have shape (). No loading methods or observations are
exposed. Output structure must not be discovered by trial execution.

## Outputs

Names are non-empty strings other than `id`. Declarations accept Field
constructor metadata, plus:

| Attribute | Meaning |
| --- | --- |
| `source` | Inherit an input's declaration. |
| `shape` | Fixed per-sample shape. |
| `sample_shapes` | Shape of each sample. |

Explicit attributes override source inheritance. New outputs require a dtype;
dimensions default to scalar. Arrays require known shapes through a fixed
shape, sample_shapes, or unchanged source structure. Changing dimensions clears
inherited coordinates unless explicitly supplied. Changing lengths requires
matching coordinates or explicit coords={}. Coordinates can reference an input
or another declared output; hidden dependencies remain attached.

Declare prediction meaning explicitly when applicable. It is not inherited
from source because arbitrary transformations may change that meaning. See the
[prediction schema](../reference/predictions.md#schema).

The matching transform must return those scalar/array values with exactly the
declared dtypes and shapes. There is no output-name parameter for analysis code.
Invalid declarations fail before computation; invalid values fail on consumption.
Ordinary missing/masked/infinite-value restrictions apply.

## Execution

| Mode | Array inputs |
| --- | --- |
| `sample` | One NumPy array per field. |
| `batch` | A tuple of arrays per field. |

Scalar inputs are native values in sample mode and NumPy vectors in batch mode.
Sample outputs contain one value per field; batch outputs are scalar vectors
or stacked/list/tuple array batches with the original sample count and order.

Field-valued parameters align and follow the same sample/batch selection.
Arrays are not stacked or padded automatically. A batch component may explicitly
stack compatible values. Sample mode can compute only a selected sample; batch
mode computes the entire originally supplied selection even for later partial
output access. Neither automatically chunks work, refits, creates windows, or
changes sample count.

Both modes are lazy and require deep-copyable state/ordinary parameters.
Component snapshots isolate later edits. Transform must not fit, depend on
changing external state, or rely on side effects. Separate accesses may recompute;
shape checks cannot detect a transform that silently reorders samples.

## Limits

This contract supports transform only. Externally fitted state is allowed, but
must be learned only from the appropriate training partition and remain fixed
while applying. Custom fitting/persistence, native Torch/TensorFlow integration,
incremental execution, and sample-creating components are not implemented.
Importable sklearn-compatible estimators use the existing native fit/predict/
save_model interface.
