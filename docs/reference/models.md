# Models
`sl.Model` connects a native PyTorch or Keras network to SignaLearn operations.
Keep architecture and training choices in the native library; SignaLearn handles
sample alignment, batch loading, and result metadata. The adapter is selected
from the native model type, including subclasses.

Install `signalearn[torch]` or `signalearn[tensorflow]` for the framework you use.
Sklearn estimators are passed directly to operations without `sl.Model`.

## PyTorch
```python
from functools import partial
import torch
import signalearn as sl

network = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, 10))
component = sl.Model(
    network,
    inputs={"image": {"dims": ("height", "width"), "dtype": "float32"}},
    task="classification",
    classes=list(range(10)),
    output="logits",
    loss=torch.nn.CrossEntropyLoss(),
    optimizer=partial(torch.optim.Adam, lr=0.001),
    epochs=5,
    batch_size=64,
)
```

Supply a `torch.nn.Module`, a scalar loss, and an optimiser factory accepting
the copied network's parameters. Training uses `train()`; inference uses
`eval()` with gradients disabled. Custom training can replace the standard loop.

## Keras
```python
from tensorflow import keras
import signalearn as sl

network = keras.Sequential([
    keras.Input((28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(10),
])
network.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
component = sl.Model(
    network,
    inputs={"image": {"dims": ("height", "width"), "dtype": "float32"}},
    task="classification",
    classes=list(range(10)),
    output="logits",
    epochs=5,
    batch_size=64,
)
```

Supply a built Keras model using the TensorFlow backend. Standard training calls
native `fit` through a batch-producing `PyDataset`. Classification requires
`SparseCategoricalCrossentropy`, with `from_logits` matching `output`.
Custom models must support Keras cloning and any required custom-object registration.

## Settings
`model`, `inputs`, and `task` are required. Native weights remain accessible as
`.model`. These components use `sl.fit`, `sl.transform`, and `sl.predict`, rather
than implementing sklearn's estimator methods themselves.

| Setting | Meaning |
| --- | --- |
| `inputs` | Named input declarations. |
| `task` | `"classification"`, `"regression"`, or `"transform"`. |
| `classes`, `output` | Ordered class labels and `"logits"` or `"probabilities"`. Classification only. |
| `outputs` | Transform field declarations. Transform only. |
| `epochs` | Training passes; default `1`. |
| `batch_size` | Samples per batch; default `64`. |
| `shuffle`, `seed` | Batch shuffling; defaults `True` and `None`. |
| `device` | Native device; defaults to CPU. |
| `trainer` | Custom training callable. |
| `loss`, `optimizer` | PyTorch loss and optimiser factory. |
| `factory`, `factory_params` | PyTorch architecture restoration. See [saving](io.md#neural-models). |

`seed` controls batch shuffling; configure native random seeds separately for
stochastic layers. Device defaults are `"cpu"` for PyTorch and `"/CPU:0"` for
Keras. PyTorch-only options are rejected for Keras; configure its training with
native `compile()`.

## Inputs
Input declarations use the [component format](../extensions/components.md#interface):
field names with `dims` and optional exact `dtype`, `unit`, and `shape` constraints.
Scalars use `dims=()`. Inputs must be finite, real numeric, fixed-shape samples.
Preprocess scaling, casting, flattening, and padding explicitly.

IDs align through the [learning input rules](learning.md#inputs). One input is
passed directly; multiple inputs become named PyTorch arguments or a Keras input
dictionary, in declaration order. Input names, dimensions, shapes, dtypes, units,
and coordinate declarations must match the fitted component.

Classification requires a scalar target and explicit ordered classes. Targets
are encoded to class indices for training. Regression requires a numeric scalar
target. Transform outputs follow the [output declarations](../extensions/components.md#outputs):
return an array for one field or a mapping for multiple fields. Frozen or stateless
transforms need no fitting; learned transforms can use a scalar supervised target.
See [Predictions](predictions.md) for label decisions, output names, and class meaning.

## Execution
`fit` returns an independent component, starting from supplied initial weights
and fresh optimiser state. These may be pretrained weights. Refitting a fitted
component restarts from its recorded initial weights; it does not resume training.
Use fresh state in each [cross-validation](evaluation.md#crossval) procedure.

Training is eager and requests batches from Dask-backed Fields. No internal
validation split is created. Inference snapshots the component and returns lazy,
batch-backed Fields. Selecting one output sample computes its batch; separate
accesses may recompute.

Required reads share upstream work within each computation. Storage chunks,
compressed files, and whole-batch transforms may require larger reads. Lazy
loading does not distribute training or guarantee bounded upstream memory.
Model artifacts and restoration are covered under [I/O](io.md#models).

## Custom training
Supply `trainer(component, batches)`. It updates the copied component's `.model`;
its return value is ignored. Each iteration yields `(inputs, target, parameters)`.
Inputs are named, targets are encoded as above, and Field-valued parameters align
by ID and compute with the batch. PyTorch yields tensors; Keras yields NumPy arrays.
`batches.shuffle()` changes sample order.

The trainer controls epochs, optimisation, and training modes. Put reusable
trainers in the [workspace extension package](../extensions/components.md#layout).
Avoid external state changes or storing observations in component declarations.
Standard PyTorch forwards extra fit parameters to the network; standard Keras
forwards ordinary parameters to native `fit`. Keras Field-valued fit parameters
and explicit validation workflows require a custom trainer.

## Scope
Supported tasks use fixed-shape numeric inputs and scalar training targets.
Classification returns one value per declared class per sample. Structured
targets, multilabel tasks, scalar binary logits, and decision scores are not
supported. Keras currently supports the TensorFlow backend. GPU execution has
not been verified, and training resumption is outside this contract.
