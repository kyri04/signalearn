from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from types import SimpleNamespace

from dask import compute, delayed
import numpy as np

from .adapters.component import declared_inputs, field_metadata, output_schema as declared_outputs
from .data import align_field, require_scalar, validate_block
from .metadata import DISPLAY_ATTRIBUTES
from .prediction import class_values, validate_prediction_values
from .preprocess import component_dataset
from .tensor import batch_task, validate_array


def validate_component(component):
    if component.task not in {"classification", "regression", "transform"}:
        raise ValueError("task must be 'classification', 'regression', or 'transform'")
    for name in ("batch_size", "epochs"):
        value = getattr(component, name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(component.shuffle) is not bool:
        raise TypeError("shuffle must be a boolean")
    if component.seed is not None and (isinstance(component.seed, bool) or not isinstance(component.seed, Integral)
                                       or component.seed < 0):
        raise ValueError("seed must be a non-negative integer or None")
    if component.trainer is not None and not callable(component.trainer):
        raise TypeError("trainer must be callable")
    if component.task == "classification":
        classes, _ = class_values(component.classes)
        if len(classes) < 2:
            raise ValueError("Classification requires at least two declared classes")
        if component.output not in {"logits", "probabilities"}:
            raise ValueError("Classification must declare output='logits' or 'probabilities'")
    elif component.classes is not None or component.output is not None:
        raise ValueError("classes and output apply only to classification")
    if component.task == "transform" and not component.outputs:
        raise ValueError("Transform components require output field declarations")
    if component.task != "transform" and component.outputs is not None:
        raise ValueError("outputs applies only to transform components")


def input_signature(fields, names):
    result = {}
    for name, field in zip(names, fields):
        metadata = field_metadata(field)
        if metadata["shape"] is None or any(size < 1 for size in metadata["shape"]):
            raise ValueError(f"Input {name!r} requires non-empty fixed-shape samples; prepare ragged data explicitly")
        if np.dtype(field.dtype).kind not in "biuf":
            raise TypeError(f"Input {name!r} must be real numeric data")
        result[name] = {key: metadata[key] for key in ("dtype", "dims", "shape", "unit", "coords")}
    return result


def fitted_inputs(component, fields, names):
    validate_component(component)
    fields, names = declared_inputs(component.inputs, fields, names)
    signature = input_signature(fields, names)
    expected = getattr(component, "_input_signature", None)
    if expected is not None and signature != expected:
        raise ValueError("Input dimensions, shapes, dtypes, units, and coordinates must match the fitted component")
    return fields, names


def fit_context(component, fields, names, target, parameters, ids):
    from ..data import Field

    fields, names = fitted_inputs(component, fields, names)
    if target is None or not isinstance(target, Field):
        raise TypeError("Neural fitting requires a supervised target Field")
    require_scalar(target, "Neural fitting targets")
    target = align_field(target, ids, allow_extra=True)
    if target.prediction is not None:
        raise ValueError("Training targets must be labels or measurements, not probability/score fields")
    if component.task == "classification":
        validate_block(np.asarray(component.classes, dtype=object), logical_dtype=target.dtype, nullable=False)
    elif target.dtype not in {"int64", "float64"}:
        raise TypeError("Regression and learned transforms require a numeric target")
    options = {name: align_field(value, ids, allow_extra=True) if isinstance(value, Field) else deepcopy(value)
               for name, value in parameters.items()}
    return fields, names, target, options


def record_fit(component, fields, names, target):
    component._input_signature = input_signature(fields, names)
    component._signalearn_inputs = tuple((name, "numeric" if field.dtype != "bool" else "bool", field.unit)
                                         for name, field in zip(names, fields))
    component._signalearn_target = {"name": target.name or "prediction", "dtype": target.dtype,
        **{key: dict(getattr(target, key)) if key in {"colours", "markers"} else getattr(target, key)
           for key in DISPLAY_ATTRIBUTES}}
    component._fitted = True


def field_task(field, indices):
    return batch_task(field, tuple(indices)) if field.dims else field._values[list(indices)]


def numeric_batch(values, field):
    result = np.stack(values) if field.dims else np.asarray(values.tolist(), dtype=field.dtype)
    if result.dtype != np.dtype(field.dtype) or result.dtype.kind not in "biuf" or not np.isfinite(result).all():
        raise ValueError(f"Input {field.name!r} contains missing, non-finite, or incompatible values")
    return np.array(result, copy=True)


@dataclass
class Batches:
    component: object
    fields: tuple
    names: tuple
    target: object = None
    parameters: object = None

    def __post_init__(self):
        self.order = np.arange(len(self.fields[0]))
        self.rng = np.random.default_rng(self.component.seed)

    def __len__(self):
        return (len(self.order) + self.component.batch_size - 1) // self.component.batch_size

    def shuffle(self):
        self.rng.shuffle(self.order)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.component.batch_size
        indices = self.order[start:start + self.component.batch_size].tolist()
        return self.read(indices)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def read(self, indices):
        from ..data import Field

        fields = dict(zip(self.names, self.fields))
        tasks = {name: field_task(field, indices) for name, field in fields.items()}
        options = self.parameters or {}
        arguments = {name: field_task(value, indices) if isinstance(value, Field) else value
                     for name, value in options.items()}
        target = field_task(self.target, indices) if self.target is not None else None
        values, labels, arguments = compute(tasks, target, arguments, scheduler="synchronous")
        values = {name: numeric_batch(value, fields[name]) for name, value in values.items()}
        arguments = {name: numeric_batch(value, options[name]) if isinstance(options[name], Field) else value
                     for name, value in arguments.items()}
        if labels is not None:
            labels = validate_block(labels, logical_dtype=self.target.dtype, nullable=False)
            if self.component.task == "classification":
                lookup = {value: index for index, value in enumerate(self.component.classes)}
                try:
                    labels = np.asarray([lookup[value] for value in labels], dtype=np.int64)
                except KeyError as error:
                    raise ValueError(f"Target label {error.args[0]!r} is absent from declared classes") from error
            else:
                labels = np.asarray(labels.tolist(), dtype=self.target.dtype)
                if not np.isfinite(labels).all():
                    raise ValueError("Training targets must be finite")
        return values, labels, arguments


def prediction_schema(component, output=None):
    validate_component(component)
    if not getattr(component, "_fitted", False):
        raise ValueError("Fit the component before prediction")
    if component.task == "transform":
        raise TypeError("Use transform for a transform component")
    metadata = deepcopy(component._signalearn_target)
    name = metadata.pop("name")
    name = "prediction" if name == "id" else name
    dtype = metadata.pop("dtype")
    if component.task == "regression":
        if output is not None:
            raise ValueError("Regression does not provide classification outputs")
        metadata.pop("colours", None)
        metadata.pop("markers", None)
        return {name: {"dtype": "float64", "nullable": False, **metadata}}
    if output is None:
        return {name: {"dtype": dtype, "nullable": False, **metadata}}
    if output not in {"probabilities", "logits"} or (output == "logits" and component.output != "logits"):
        raise ValueError("Classification supports probabilities and explicitly declared logits; scores are not inferred")
    classes = list(component.classes)
    suffix = "probability" if output == "probabilities" else "logit"
    meaning = {"kind": output, "task": "binary" if len(classes) == 2 else "multiclass",
               "classes": classes, "target": name,
               "class_metadata": {key: metadata[key] for key in ("label", "label_latex", "colours", "markers")}}
    return {f"{name}_{suffix}": {"dtype": "float64", "nullable": False, "dims": ("class",),
        "shape": (len(classes),), "coords": {"class": classes}, "prediction": meaning,
        "label": f"{metadata.get('label') or name} {suffix}", "colour": metadata["colour"], "marker": metadata["marker"]}}


def output_schema(component, fields, names):
    if component.task != "transform":
        raise TypeError("Use predict for a classification or regression component")
    return declared_outputs(SimpleNamespace(signalearn_outputs=lambda inputs: component.outputs), fields, names)


def prediction_result(component, raw, schema, output, length):
    if np.ma.isMaskedArray(raw):
        raise ValueError("Masked model outputs are unsupported")
    raw = np.asarray(raw)
    if raw.dtype.kind not in "biuf" or not np.isfinite(raw).all():
        raise ValueError("Model output must be finite real numeric values")
    name, attributes = next(iter(schema.items()))
    if component.task == "regression":
        if raw.shape not in {(length,), (length, 1)}:
            raise ValueError("Regression must return one value per sample")
        from .adapters.sklearn import numeric_output

        result = numeric_output(raw, (length, 1), "Prediction").ravel()
    else:
        if raw.shape != (length, len(component.classes)):
            raise ValueError("Classification must return one value per declared class and sample")
        values = raw.astype(np.float64)
        if component.output == "probabilities":
            meaning = {"kind": "probabilities", "task": "binary" if len(component.classes) == 2 else "multiclass"}
            validate_prediction_values(values, meaning, ("class",), batched=True)
        if output is None:
            result = np.asarray(component.classes, dtype=object)[values.argmax(axis=1)]
        elif output == "probabilities" and component.output == "logits":
            exp = np.exp(values - values.max(axis=1, keepdims=True))
            result = exp / exp.sum(axis=1, keepdims=True)
        else:
            result = values
        if output is not None:
            result = validate_prediction_values(result, attributes["prediction"], ("class",), batched=True)
    return {name: result}


def validated_transform(raw, schema, indices):
    if not isinstance(raw, Mapping):
        if len(schema) != 1:
            raise ValueError("Multiple transform outputs require a mapping with declared names")
        raw = {next(iter(schema)): raw}
    if set(raw) != set(schema):
        raise ValueError("Transform must return exactly the declared output fields")
    result = {}
    for name, specification in schema.items():
        values = raw[name]
        if len(values) != len(indices):
            raise ValueError("Transform output must preserve sample count and order")
        if specification["attributes"]["dims"]:
            result[name] = tuple(validate_array(value, specification["dtype"], specification["nullable"],
                                               specification["shapes"][index], name)
                                 for index, value in zip(indices, values))
        else:
            result[name] = validate_block(values, logical_dtype=specification["dtype"], nullable=specification["nullable"])
    return result


def sample_result(result, index):
    return {name: value[index] for name, value in result.items()}


def deferred_output(adapter, model, fields, names, ids, schema, output, parameters):
    from ..data import Field

    model = adapter.snapshot(model)
    options = {key: align_field(value, ids, allow_extra=True) if isinstance(value, Field) else deepcopy(value)
               for key, value in parameters.items()}
    if output != "transform":
        schema = declared_outputs(SimpleNamespace(signalearn_outputs=lambda inputs: schema), fields, names)
    reader = Batches(model, fields, names, parameters=options)
    tasks = []
    for start in range(0, len(ids), model.batch_size):
        indices = tuple(range(start, min(start + model.batch_size, len(ids))))
        task = delayed(adapter.apply_batch)(delayed(model, traverse=False), delayed(reader, traverse=False),
                                           indices, delayed(schema, traverse=False), output)
        tasks.extend(delayed(sample_result)(task, index) for index in range(len(indices)))
    return component_dataset(tasks, schema, fields, names, ids, False)
