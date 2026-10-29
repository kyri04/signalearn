from collections.abc import Mapping
from copy import copy, deepcopy
from functools import partial
from numbers import Integral, Real
import re
from types import BuiltinFunctionType, FunctionType, MethodType

import joblib
import numpy as np
from scipy import sparse
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

from ..data import infer_dtype, require_scalar, validate_block
from ..metadata import DISPLAY_ATTRIBUTES
from ..model_io import importable_reference
from ..prediction import class_values, validate_prediction_values
from ..features import feature_contract, feature_names, feature_width, preserving_transform

persistence_name = "sklearn"
persistence_format = "joblib"
persistence_payload = "model.joblib"
persistence_dependencies = ("scikit-learn", "numpy", "scipy", "joblib")


def supports(model, operation):
    return (not isinstance(model, type) and callable(getattr(model, "get_params", None))
            and callable(getattr(model, "fit", None))
            and (operation == "save_model" or callable(getattr(model, operation, None))))


def validate_model_parameters(value, seen):
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, (FunctionType, BuiltinFunctionType, type)):
        importable_reference(value)
    elif isinstance(value, MethodType):
        importable_reference(value.__func__)
        validate_model_parameters(value.__self__, seen)
    elif isinstance(value, partial):
        validate_model_parameters((value.func, value.args, value.keywords), seen)
    elif isinstance(value, Mapping):
        for item in value.values():
            validate_model_parameters(item, seen)
    elif isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            validate_model_parameters(item, seen)
    elif callable(getattr(value, "get_params", None)):
        importable_reference(type(value))
        validate_model_parameters(value.get_params(deep=False), seen)
    elif callable(value):
        reference = value if getattr(value, "__module__", None) and getattr(value, "__qualname__", None) else type(value)
        importable_reference(reference)


def validate_persistence(model):
    check_is_fitted(model)
    importable_reference(type(model))
    validate_model_parameters(model.get_params(deep=True), set())


def persistence_metadata(model):
    inputs = getattr(model, "_signalearn_inputs", None)
    vectors = getattr(model, "_signalearn_vectors", {})
    return {"inputs": None if inputs is None else [{"name": name, "kind": kind, "unit": unit,
                                                   **({"vector": vectors[name]} if name in vectors else {})}
                                                  for name, kind, unit in inputs],
            "target": getattr(model, "_signalearn_target", None)}


def dump_model(model, path):
    stored = copy(model)
    if stored is model:
        raise TypeError("Component copying must return an independent instance for persistence")
    for attribute in ("_signalearn_inputs", "_signalearn_target", "_signalearn_vectors"):
        if hasattr(stored, attribute):
            delattr(stored, attribute)
    try:
        joblib.dump(stored, path, protocol=5)
    except Exception as error:
        error.add_note("Could not serialize the component; custom classes and functions must be importable")
        raise


def load_model(path):
    try:
        return joblib.load(path)
    except ImportError as error:
        raise ImportError("Cannot restore model: install its required libraries and make its custom component "
                          f"modules importable. Original error: {error}") from error
    except AttributeError as error:
        raise AttributeError("Cannot restore model: a saved class or function is unavailable in its original "
                             f"module. Original error: {error}") from error
    except Exception as error:
        error.add_note(f"Could not restore native model payload: {path}")
        raise


def restore_metadata(model, metadata):
    for attribute in ("_signalearn_inputs", "_signalearn_target", "_signalearn_vectors"):
        if hasattr(model, attribute):
            delattr(model, attribute)
    if metadata["inputs"] is not None:
        model._signalearn_inputs = tuple((field["name"], field["kind"], field["unit"])
                                         for field in metadata["inputs"])
        model._signalearn_vectors = {field["name"]: deepcopy(field["vector"]) for field in metadata["inputs"]
                                     if "vector" in field}
    if metadata["target"] is not None:
        model._signalearn_target = deepcopy(metadata["target"])


def input_kind(field):
    return "numeric" if (field.dims and field.dtype != "bool") or field.dtype in {"int64", "float64"} else field.dtype


def fit_model(model, matrix, target, parameters, fields, names, target_field=None):
    fitted = clone(model)
    if target is None:
        result = fitted.fit(matrix, **parameters)
    else:
        result = fitted.fit(matrix, target, **parameters)
    if result is not fitted:
        raise TypeError("A scikit-learn compatible fit method must return the fitted component itself")
    fitted._signalearn_inputs = tuple((name, input_kind(field), field.unit)
                                     for name, field in zip(names, fields))
    if target_field is not None:
        fitted._signalearn_target = {
            "name": target_field.name or "prediction", "dtype": target_field.dtype,
            **{key: dict(getattr(target_field, key)) if key in {"colours", "markers"}
               else getattr(target_field, key) for key in DISPLAY_ATTRIBUTES},
        }
    return fitted


def fit_fields(model, fields, names, target, parameters, ids):
    from ..alignment import model_inputs

    feature_names(fields, names)
    contract = feature_contract(fields, names)
    matrix, labels, options = model_inputs(fields, target, parameters, ids).compute()
    fitted = fit_model(model, matrix, labels, options, fields, names, target)
    fitted._signalearn_vectors = contract
    return fitted


def fitted_inputs(model, fields, names):
    check_is_fitted(model)
    for field in fields:
        feature_width(field)
    schema = getattr(model, "_signalearn_inputs", None)
    expected = tuple(item[0] for item in schema) if schema is not None else getattr(model, "feature_names_in_", None)
    if schema is None and any(field.dims for field in fields):
        if expected is not None and tuple(expected) != feature_names(fields, names):
            raise ValueError("Expanded vector feature names must match the externally fitted component's feature order")
        expected = None
    if expected is not None:
        expected = tuple(expected)
        if set(expected) != set(names) or len(expected) != len(names):
            raise ValueError(f"Input fields must match the fitted component's fields: {expected!r}")
        lookup = dict(zip(names, fields))
        fields, names = tuple(lookup[name] for name in expected), expected
    if schema is not None:
        if feature_contract(fields, names) != getattr(model, "_signalearn_vectors", {}):
            raise ValueError("Vector feature dimensions, lengths, and coordinates must match the fitted component")
        for field, (name, kind, unit) in zip(fields, schema):
            if input_kind(field) != kind:
                raise ValueError(f"Field {name!r} has an incompatible logical dtype")
            if field.unit != unit:
                raise ValueError(f"Field {name!r} must use the fitted unit {unit!r}; no unit conversion is inferred")
    else:
        feature_contract(fields, names)
    count = len(feature_names(fields, names))
    if getattr(model, "n_features_in_", count) != count:
        raise ValueError("The number of input features must match the fitted component")
    return fields, names


def output_features(model, input_names):
    method = getattr(model, "get_feature_names_out", None)
    output_names = None
    if callable(method):
        try:
            output_names = method(list(input_names))
        except (AttributeError, NotImplementedError):
            pass
    if output_names is None and type(model) is FunctionTransformer and model.func is None:
        output_names = input_names
    derived = output_names is None
    if output_names is None:
        count = getattr(model, "n_features_out_", None)
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count < 1:
            raise ValueError("The component or its adapter must declare output structure through "
                             "get_feature_names_out or a positive n_features_out_")
        prefix = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", type(model).__name__)
        prefix = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", prefix).strip("_").lower()
        output_names = ((f"{prefix}_{input_names[0]}",) if count == 1 and len(input_names) == 1
                        else tuple(f"{prefix}_{index}" for index in range(count)))
    if isinstance(output_names, str):
        output_names = (output_names,)
    names = tuple(output_names)
    if not names or any(not isinstance(name, str) or not name or name == "id" for name in names):
        raise ValueError("Output names must be non-empty strings other than 'id'")
    if len(set(names)) != len(names):
        raise ValueError("Output names must be unique")
    return tuple((name, name if not derived and name in input_names else None) for name in names)


def output_schema(model, fields, names):
    sources = dict(zip(names, fields))
    if any(field.dims for field in fields) and preserving_transform(model):
        return {name: {"dtype": "float64", "nullable": True, **output_metadata(model, field),
                        **({"dims": field.dims, "shape": (feature_width(field),),
                            "coords": {dim: {"field": coordinate.name} if hasattr(coordinate, "ids") else coordinate.tolist()
                                       for dim, coordinate in field.coords.items()}} if field.dims else {})}
                for field, name in zip(fields, names)}
    return {name: {"dtype": "float64", "nullable": True,
                   **output_metadata(model, sources.get(source))}
            for name, source in output_features(model, feature_names(fields, names))}


def transform_shape(schema):
    return sum(attributes["shape"][0] if attributes.get("dims") else 1 for attributes in schema.values())


def preserves_units(model):
    return ((type(model) is StandardScaler and not model.with_std)
            or (type(model) is RobustScaler and not model.with_scaling)
            or (type(model) is FunctionTransformer and model.func is None))


def output_metadata(model, source):
    if source is None:
        return {}
    metadata = {name: getattr(source, name) for name in ("label", "label_latex", "colour", "marker")}
    if preserves_units(model):
        metadata.update(unit=source.unit, unit_latex=source.unit_latex)
    return metadata


def snapshot(model):
    return deepcopy(model)


def apply_transform(model, matrix, parameters, shape):
    result = model.transform(matrix, **parameters)
    return numeric_output(result, shape, "Transform")


def numeric_output(result, shape, operation):
    if sparse.issparse(result):
        result = result.toarray()
    result = np.asarray(result)
    if result.ndim == 1:
        result = result.reshape(-1, 1)
    if result.ndim != 2 or result.shape != shape:
        raise ValueError(f"{operation} returned shape {result.shape!r}; expected {shape!r} with unchanged sample count")
    if result.dtype.kind not in "biuf":
        if result.dtype.kind != "O" or any(value is not None and not isinstance(value, Real) for value in result.flat):
            raise TypeError(f"{operation} requires real numeric output fields")
    if result.dtype.kind in "iuO":
        for value in result.flat:
            if isinstance(value, (int, np.integer)) and int(float(value)) != int(value):
                raise ValueError(f"{operation} output cannot be represented as float64 without losing integer precision")
    result = np.asarray(result, dtype=float)
    if np.isinf(result).any():
        raise ValueError(f"{operation} output contains infinite values")
    return result


def prediction_classes(model):
    classes = np.asarray(getattr(model, "classes_", None), dtype=object)
    if classes.ndim != 1 or not len(classes):
        raise ValueError("Prediction requires a single classification target with declared classes_")
    return classes


def prediction_schema(model, output=None):
    if output is not None:
        return classification_output_schema(model, output)
    metadata = deepcopy(getattr(model, "_signalearn_target", {}))
    name = metadata.pop("name", "prediction")
    dtype = metadata.pop("dtype", None)
    if name == "id":
        name = "prediction"
    if is_classifier(model):
        classes = prediction_classes(model)
        dtype = infer_dtype(classes) if dtype is None else dtype
        validate_block(classes, logical_dtype=dtype, nullable=False)
    elif is_regressor(model):
        if getattr(model, "n_outputs_", 1) != 1:
            raise ValueError("Prediction currently supports a single regression target")
        dtype = "float64"
        metadata.pop("colours", None)
        metadata.pop("markers", None)
    else:
        raise TypeError("Prediction currently supports scikit-learn classifiers and regressors")
    return {name: {"dtype": dtype, "nullable": False, **metadata}}


def apply_prediction(model, matrix, parameters, length, schema, output=None):
    if output is not None:
        return classification_output(model, matrix, parameters, length, schema, output)
    result = model.predict(matrix, **parameters)
    attributes = next(iter(schema.values()))
    if is_regressor(model):
        result = numeric_output(result, (length, 1), "Prediction")
        if np.isnan(result).any():
            raise ValueError("Prediction output must not contain missing values")
        return result
    result = np.asarray(result)
    if result.ndim == 2 and result.shape == (length, 1):
        result = result[:, 0]
    if result.shape != (length,):
        raise ValueError(f"Prediction returned shape {result.shape!r}; expected {(length,)!r} with unchanged sample count")
    result = validate_block(result, logical_dtype=attributes["dtype"], nullable=False)
    classes = set(prediction_classes(model).tolist())
    if any(value not in classes for value in result.tolist()):
        raise ValueError("Prediction output contains labels absent from the fitted classes_")
    return result.reshape(-1, 1)


def decision_component(model):
    seen = set()
    while id(model) not in seen:
        seen.add(id(model))
        nested = getattr(model, "best_estimator_", getattr(model, "_final_estimator", None))
        if nested is None:
            return model
        model = nested
    raise ValueError("Cyclic classifier composition")


def classification_output_schema(model, output):
    if output not in {"probabilities", "scores"}:
        raise ValueError("output must be None, 'probabilities', or 'scores'")
    if not is_classifier(model):
        raise TypeError(f"{output} requires a classifier")
    component = decision_component(model)
    if (getattr(component, "multilabel_", False)
            or getattr(getattr(component, "_label_binarizer", None), "y_type_", None) == "multilabel-indicator"):
        raise ValueError("This integration supports a single classification target, not multilabel outputs")
    classes, _ = class_values(prediction_classes(model))
    metadata = deepcopy(getattr(model, "_signalearn_target", {}))
    if "dtype" in metadata:
        validate_block(np.asarray(classes, dtype=object), logical_dtype=metadata["dtype"], nullable=False)
    name = metadata.get("name", "prediction")
    if name == "id":
        name = "prediction"
    native_method = "predict_proba" if output == "probabilities" else "decision_function"
    if not callable(getattr(model, native_method, None)):
        raise TypeError(f"{type(model).__name__} does not support {output}; the fitted model must expose {native_method}")
    if output == "scores":
        if len(classes) < 2:
            raise ValueError("Decision scores require at least two classes")
        if len(classes) > 2 and getattr(component, "decision_function_shape", None) == "ovo":
            raise ValueError("One-vs-one decision scores need a class-pair representation; use per-class scores instead")
    binary_score = output == "scores" and len(classes) == 2
    meaning = {
        "kind": "probabilities" if output == "probabilities" else "decision_scores",
        "task": "binary" if len(classes) == 2 else "multiclass",
        "target": name, "classes": classes,
        "class_metadata": {key: metadata[key] for key in ("label", "label_latex", "colours", "markers")
                           if key in metadata},
    }
    if binary_score:
        meaning["positive_class"] = classes[1]
    suffix = "probability" if output == "probabilities" else "score"
    attributes = {"dtype": "float64", "nullable": False, "prediction": meaning,
                  "label": f"{metadata.get('label') or name} {suffix}",
                  **{key: metadata[key] for key in ("colour", "marker") if key in metadata}}
    if not binary_score:
        attributes.update(dims=("class",), shape=(len(classes),), coords={"class": classes})
    return {f"{name}_{suffix}": attributes}


def classification_output(model, matrix, parameters, length, schema, output):
    attributes = next(iter(schema.values()))
    method = model.predict_proba if output == "probabilities" else model.decision_function
    result = method(matrix, **parameters)
    if np.ma.isMaskedArray(result):
        raise ValueError(f"{output} masked outputs require an explicit supported mask representation")
    width = attributes["shape"][0] if attributes.get("dims") else 1
    expected = (length, width) if attributes.get("dims") else (length,)
    if np.shape(result) != expected:
        raise ValueError(f"{output} returned shape {np.shape(result)!r}; expected {expected!r} from the native class contract")
    result = numeric_output(result, (length, width), output)
    return validate_prediction_values(result, attributes["prediction"], attributes.get("dims", ()), batched=True)
