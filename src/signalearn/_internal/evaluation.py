from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
import warnings

from dask import compute
import numpy as np

from ..data import Dataset, Field
from .data import align_field, clone_field, new_ids, require_scalar, validate_block
from .metadata import DISPLAY_ATTRIBUTES


def metric_fields(predictions, target, operation, *, numeric=False):
    if isinstance(predictions, Dataset):
        fields = [field for name, field in predictions.fields.items() if name != "id"]
        if len(fields) != 1:
            raise ValueError(f"{operation} requires exactly one prediction field; select it explicitly")
        predictions = fields[0]
    if not isinstance(predictions, Field) or not isinstance(target, Field):
        raise TypeError(f"{operation} requires a prediction Field or single-output Dataset and a target Field")
    if predictions.prediction is not None or target.prediction is not None:
        raise ValueError(f"{operation} requires labels or regression values, not probabilities or decision scores")
    require_scalar(predictions, f"{operation} predictions")
    require_scalar(target, f"{operation} targets")
    if not len(predictions):
        raise ValueError(f"{operation} requires at least one sample")
    numeric_types = {"int64", "float64"}
    if numeric and not {predictions.dtype, target.dtype} <= numeric_types:
        raise TypeError(f"{operation} requires numeric predictions and targets")
    if predictions.dtype != target.dtype and not {predictions.dtype, target.dtype} <= numeric_types:
        raise ValueError("Prediction and target labels must have compatible logical dtypes")
    if predictions.unit != target.unit:
        raise ValueError("Prediction and target units must match; no unit conversion is inferred")
    return predictions, align_field(target, predictions.ids)


def validate_metric_values(predictions, target, operation):
    if any(value is None for values in (predictions, target) for value in values):
        raise ValueError(f"{operation} requires complete predictions and targets; missing values are not dropped")


@dataclass(frozen=True)
class MetricGroups:
    field: Field
    labels: tuple
    pairs: tuple
    counts: tuple
    target: Field


def summary_field(field, values, ids):
    import dask.array as da

    values = np.array(values, copy=True)
    values.flags.writeable = False
    return clone_field(field, storage=da.from_array(values, chunks=1024), ids=ids)


def metric_groups(predictions, target, by, operation, *, numeric=False, classification=None):
    predictions, target = metric_fields(predictions, target, operation, numeric=numeric)
    if classification is not None:
        classification_options(*classification, target.dtype)
    if not isinstance(by, Field):
        raise TypeError("by must be a SignaLearn Field")
    require_scalar(by, "Evaluation grouping")
    by = align_field(by, predictions.ids)
    predicted, expected, values = compute(predictions._values, target._values, by._values)
    validate_metric_values(predicted, expected, operation)
    if any(value is None for value in values):
        raise ValueError("Evaluation grouping requires complete values; missing groups are not dropped")
    predicted = summary_field(predictions, predicted, predictions.ids)
    expected = summary_field(target, expected, predictions.ids)
    membership = {}
    for index, value in enumerate(values.tolist()):
        membership.setdefault(value, []).append(index)
    labels = tuple(sorted(membership))
    pairs = tuple((predicted[membership[value]], expected[membership[value]]) for value in labels)
    return MetricGroups(by, labels, pairs, tuple(len(membership[value]) for value in labels), target)


def grouped_result(groups, scores, name):
    from ..preprocess import concat

    group_name = groups.field.name or "group"
    if group_name in {"id", "category", "support", "count", name}:
        group_name += "_group"
    if isinstance(scores[0], Dataset):
        result = concat([score.select(name, "support") for score in scores])
        categories = tuple(value for score in scores for value in score.category.values.tolist())
        values = [label for label, score in zip(groups.labels, scores) for _ in range(len(score))]
        group = summary_field(groups.field, values, result.ids)
        category = category_values(categories, groups.target, result.ids)
        return result.assign({group_name: group, "category": category}).select(group_name, "category", name, "support")
    ids = new_ids(len(scores))
    metadata = {"label": name.upper() if name in {"mae", "rmse", "r2"} else name.capitalize()}
    if name in {"mae", "rmse"}:
        metadata.update(unit=groups.target.unit, unit_latex=groups.target.unit_latex)
    return Dataset({
        group_name: summary_field(groups.field, groups.labels, ids),
        name: Field([float(score) if np.isfinite(score) else None for score in scores],
                    ids=ids, dtype="float64", **metadata),
        "count": Field(groups.counts, ids=ids, dtype="int64", label="Sample count"),
    })


@dataclass(frozen=True)
class ClassificationData:
    target: Field
    labels: tuple
    predicted: np.ndarray
    expected: np.ndarray
    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray
    tn: np.ndarray


def classification_options(average, positive, dtype):
    if average is not None and (not isinstance(average, str) or average not in {"binary", "micro", "macro", "weighted"}):
        raise ValueError("average must be 'binary', 'micro', 'macro', 'weighted', or None")
    if average == "binary":
        if positive is None:
            raise ValueError("Binary metrics require an explicit positive class")
        try:
            validate_block(np.array([positive], dtype=object), logical_dtype=dtype, nullable=False)
        except (TypeError, ValueError) as error:
            raise ValueError("positive must be a finite class label compatible with the target dtype") from error
    elif positive is not None:
        raise ValueError("positive is only applicable with average='binary'")


def classification_data(predictions, target, average=None, positive=None):
    predictions, target = metric_fields(predictions, target, "Classification evaluation")
    classification_options(average, positive, target.dtype)
    predicted, expected = compute(predictions._values, target._values)
    validate_metric_values(predicted, expected, "Classification evaluation")
    predicted, expected = predicted.tolist(), expected.tolist()
    labels = set(expected) | set(predicted)
    if average == "binary":
        labels.add(positive.item() if isinstance(positive, np.generic) else positive)
        if len(labels) > 2:
            raise ValueError("Binary averaging requires at most two classes, including the positive class")
    labels = tuple(sorted(labels))
    lookup = {value: index for index, value in enumerate(labels)}
    predicted = np.array([lookup[value] for value in predicted], dtype=np.intp)
    expected = np.array([lookup[value] for value in expected], dtype=np.intp)
    tp = np.bincount(expected[predicted == expected], minlength=len(labels))
    fp = np.bincount(predicted, minlength=len(labels)) - tp
    fn = np.bincount(expected, minlength=len(labels)) - tp
    tn = len(expected) - tp - fp - fn
    return ClassificationData(target, labels, predicted, expected, tp, fp, fn, tn)


def metric_division(numerator, denominator, name):
    denominator = np.asarray(denominator)
    invalid = denominator == 0
    if invalid.any():
        warnings.warn(f"{name} is undefined for {int(invalid.sum())} score(s) with a zero denominator; returning NaN",
                      UserWarning, stacklevel=3)
    return np.divide(numerator, denominator, out=np.full(denominator.shape, np.nan), where=~invalid)


def category_field(data, ids):
    return category_values(data.labels, data.target, ids)


def category_values(labels, target, ids):
    dtype = target.dtype
    if dtype in {"int64", "float64"}:
        if all(isinstance(value, Integral) for value in labels):
            dtype = "int64"
        else:
            if any(float(value) != value for value in labels):
                raise ValueError("Class labels cannot share a numeric Field without losing precision; use matching label dtypes")
            dtype = "float64"
    options = {key: getattr(target, key) for key in DISPLAY_ATTRIBUTES}
    if options["label"] is None:
        options["label"] = target.name or "Class"
    return Field(labels, ids=ids, dtype=dtype, **options)


def classification_result(data, numerator, denominator, average, positive, name):
    support = data.tp + data.fn
    if average == "binary":
        index = data.labels.index(positive)
        return float(metric_division(numerator[index], denominator[index], name))
    if average == "micro":
        return float(metric_division(numerator.sum(), denominator.sum(), name))
    if average == "weighted":
        present = support > 0
        scores = metric_division(numerator[present], denominator[present], name)
        return float(np.average(scores, weights=support[present]))
    scores = metric_division(numerator, denominator, name)
    if average == "macro":
        return float(np.mean(scores))
    ids = new_ids(len(data.labels))
    return Dataset({
        "category": category_field(data, ids),
        name: Field([None if np.isnan(value) else float(value) for value in scores],
                    ids=ids, dtype="float64", label="F1" if name == "f1" else name.capitalize()),
        "support": Field(support, ids=ids, dtype="int64", label="Support"),
    })


def numeric_differences(left, right):
    result = []
    for value, reference in zip(left, right):
        if isinstance(value, Integral) != isinstance(reference, Integral):
            difference = Fraction(value) - Fraction(reference)
        else:
            difference = value - reference
        try:
            difference = float(difference)
        except OverflowError as error:
            raise ValueError("Regression differences exceed the float64 range") from error
        if not np.isfinite(difference):
            raise ValueError("Regression differences exceed the float64 range")
        result.append(difference)
    return np.array(result)


def regression_data(predictions, target, name):
    predictions, target = metric_fields(predictions, target, name, numeric=True)
    predicted, expected = compute(predictions._values, target._values)
    validate_metric_values(predicted, expected, name)
    predicted, expected = predicted.tolist(), expected.tolist()
    return numeric_differences(predicted, expected), expected


def confusion_values(data, normalize):
    matrix = np.zeros((len(data.labels), len(data.labels)), dtype=np.int64)
    np.add.at(matrix, (data.expected, data.predicted), 1)
    if normalize is None:
        return matrix
    denominator = (matrix.sum(axis=1, keepdims=True) if normalize == "true"
                   else matrix.sum(axis=0, keepdims=True) if normalize == "pred" else matrix.sum())
    if np.any(denominator == 0):
        warnings.warn("Confusion matrix normalization is undefined for empty classes; cells are shown as missing",
                      UserWarning, stacklevel=3)
    return np.divide(matrix, denominator, out=np.full(matrix.shape, np.nan), where=denominator != 0)
