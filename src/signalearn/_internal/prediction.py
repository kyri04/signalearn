from collections.abc import Mapping, Sequence
from types import MappingProxyType

import numpy as np

from .metadata import prepare_colours, prepare_markers


def class_values(values):
    from .data import infer_dtype, validate_block

    if isinstance(values, (str, bytes)) or not isinstance(values, (Sequence, np.ndarray)):
        raise ValueError("Prediction classes must be a non-empty sequence of labels")
    values = tuple(value.item() if isinstance(value, np.generic) else value for value in values)
    if not values or any(not isinstance(value, (str, bool, int, float)) for value in values):
        raise ValueError("Prediction classes must contain scalar labels")
    dtype = infer_dtype(np.asarray(values, dtype=object))
    validate_block(np.asarray(values, dtype=object), logical_dtype=dtype, nullable=False)
    if len(set(values)) != len(values):
        raise ValueError("Prediction classes must be unique")
    return values, dtype


def class_metadata(metadata, dtype, *, encoded=False):
    if not isinstance(metadata, Mapping) or set(metadata) - {"label", "label_latex", "colours", "markers"}:
        raise ValueError("Prediction class_metadata supports label, label_latex, colours, and markers")
    result = {}
    for name in ("label", "label_latex"):
        value = metadata.get(name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Prediction class {name} must be a string or None")
        result[name] = value
    for name, prepare in (("colours", prepare_colours), ("markers", prepare_markers)):
        values = metadata.get(name, [] if encoded else {})
        if encoded:
            if (not isinstance(values, list) or any(not isinstance(pair, list) or len(pair) != 2
                                                   or not isinstance(pair[0], (str, bool, int, float))
                                                   for pair in values)):
                raise ValueError(f"Prediction class {name} must contain category/display pairs")
            if len({pair[0] for pair in values}) != len(values):
                raise ValueError(f"Prediction class {name} contains repeated categories")
            values = dict(values)
        result[name] = prepare(values, dtype)
    return MappingProxyType(result)


def prepare_prediction(metadata, dims=(), coords=None, *, dtype=None, shape=None, nullable=False, encoded=False):
    if metadata is None:
        return None
    required = {"kind", "task", "target", "classes"}
    if (not isinstance(metadata, Mapping) or not required <= metadata.keys()
            or metadata.keys() - required - {"positive_class", "class_metadata"}):
        raise ValueError("Prediction metadata requires kind, task, target, classes, and supported attributes only")
    result = dict(metadata)
    if nullable:
        raise ValueError("Prediction fields must not be nullable")
    if isinstance(result.get("positive_class"), np.generic):
        result["positive_class"] = result["positive_class"].item()
    if result["kind"] not in {"probabilities", "decision_scores", "logits"}:
        raise ValueError("Unsupported prediction kind")
    if result["task"] not in {"binary", "multiclass", "multilabel"}:
        raise ValueError("Unsupported prediction task")
    if not isinstance(result["target"], str) or not result["target"]:
        raise ValueError("Prediction target must be a non-empty name")
    classes, class_dtype = class_values(result["classes"])
    if result["task"] == "binary" and len(classes) != 2:
        raise ValueError("Binary predictions require exactly two classes")
    if dtype is not None and not dtype.startswith("float"):
        raise ValueError("Probabilities and scores require floating-point fields")
    if "class" in dims:
        coordinate = (coords or {}).get("class")
        if isinstance(coordinate, (str, bytes, Mapping)) or coordinate is None:
            raise ValueError("Predictions require inline class coordinates")
        labels, coordinate_dtype = class_values(coordinate)
        if labels != classes or coordinate_dtype != class_dtype:
            raise ValueError("Prediction class coordinates must match declared classes in order and type")
        if shape is not None and shape[tuple(dims).index("class")] not in {None, len(classes)}:
            raise ValueError("Prediction class shape must match declared classes")
        if "positive_class" in result:
            raise ValueError("Per-class predictions do not take a positive_class")
    elif (result["task"] != "binary" or result["kind"] == "probabilities"
          or "positive_class" not in result):
        raise ValueError("Predictions require a class dimension; a binary score may declare positive_class instead")
    elif (result["positive_class"] not in classes
          or type(result["positive_class"]) is not type(classes[classes.index(result["positive_class"])])):
        raise ValueError("Prediction positive_class must be one of the declared class labels")
    result["classes"] = classes
    result["class_metadata"] = class_metadata(result.get("class_metadata", {}), class_dtype, encoded=encoded)
    return MappingProxyType(result)


def encoded_prediction(metadata):
    if metadata is None:
        return None
    result = dict(metadata)
    result["classes"] = list(result["classes"])
    display = dict(result["class_metadata"])
    for name in ("colours", "markers"):
        display[name] = [[key.item() if isinstance(key, np.generic) else key, value]
                         for key, value in display[name].items()]
    result["class_metadata"] = display
    return result


def validate_prediction_values(value, metadata, dims, *, batched=False):
    if not np.isfinite(value).all():
        raise ValueError("Predictions must not contain missing or infinite values")
    if metadata["kind"] == "probabilities":
        if (value < 0).any() or (value > 1).any():
            raise ValueError("Probabilities must be between zero and one")
        axis = tuple(dims).index("class") + int(batched)
        if metadata["task"] != "multilabel" and not np.allclose(value.sum(axis=axis), 1, rtol=1e-6, atol=1e-8):
            raise ValueError("Exclusive-class probabilities must sum to one per sample")
    return value


def validate_prediction_batch(values, metadata, dims):
    return tuple(validate_prediction_values(value, metadata, dims) for value in values)


def aligned_prediction_fields(fields, name):
    from .data import clone_field
    from .tensor import ArrayStorage, IndexedArray

    first = fields[0]
    aligned = []
    for field in fields:
        if (field.prediction == first.prediction and (field.prediction is None
                or class_values(field.prediction["classes"])[1] == class_values(first.prediction["classes"])[1])):
            aligned.append(field)
            continue
        if field.prediction is None or first.prediction is None:
            raise ValueError(f"Field {name!r}: prediction metadata must match")
        classes, desired = field.prediction["classes"], first.prediction["classes"]
        if ("class" not in field.dims or "class" not in first.dims or set(classes) != set(desired)
                or class_values(classes)[1] != class_values(desired)[1]):
            raise ValueError(f"Field {name!r}: prediction classes must match; missing classes are not filled")
        metadata = dict(field.prediction, classes=desired)
        if metadata != first.prediction:
            raise ValueError(f"Field {name!r}: incompatible prediction meaning or class metadata")
        axis = field.dims.index("class")
        order = tuple(classes.index(label) for label in desired)
        storage = ArrayStorage(tuple(IndexedArray(item, ((axis, order),), item.shape, item.dtype)
                                     for item in field._values.items), field._values.dtype,
                               field._values.sample_shape)
        result = clone_field(field, storage=storage)
        object.__setattr__(result, "_coords", MappingProxyType(dict(field._coords, **{"class": desired})))
        object.__setattr__(result, "prediction", MappingProxyType(metadata))
        aligned.append(result)
    return aligned
