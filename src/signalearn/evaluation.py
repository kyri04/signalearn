import warnings
from collections.abc import Callable
from copy import deepcopy

from dask import compute
import numpy as np

from ._internal.evaluation import (
    classification_data, classification_result, grouped_result, metric_fields, metric_groups, numeric_differences,
    regression_data, validate_metric_values,
)
from ._internal.crossval import crossval_inputs, prediction_output, reference_output, snapshot_predictions
from .data import Dataset, Field
from .preprocess import Folds, concat

__all__ = ["crossval", "accuracy", "precision", "recall", "sensitivity", "specificity", "f1", "mae", "rmse", "r2"]


def crossval(
    splits: Folds, procedure: Callable[[Dataset, Dataset], Dataset], *, target: Field | None = None,
) -> Dataset:
    initial, target, rows, ids, repeated = crossval_inputs(splits, procedure, target)
    results, offset = [], 0
    for number, (train, test) in enumerate(splits):
        try:
            output = deepcopy(initial)(train, test)
            output = prediction_output(output, test)
            output = reference_output(output, target, splits.indices[number][1])
            results.append(snapshot_predictions(output, ids[offset:offset + len(test)], number))
        except Exception as error:
            error.add_note(f"Cross-validation failed in split {number}")
            raise
        offset += len(test)
    result = concat(results)
    if not repeated:
        result = result[sorted(range(len(rows)), key=rows.__getitem__)]
    return result


def accuracy(predictions: Field | Dataset, target: Field, *, by: Field | None = None) -> float | Dataset:
    if by is not None:
        groups = metric_groups(predictions, target, by, "Accuracy")
        scores = [accuracy(predicted, expected) for predicted, expected in groups.pairs]
        return grouped_result(groups, scores, "accuracy")
    predictions, target = metric_fields(predictions, target, "Accuracy")
    predicted, expected = compute(predictions._values, target._values)
    validate_metric_values(predicted, expected, "Accuracy")
    matches = sum(left == right for left, right in zip(predicted.tolist(), expected.tolist()))
    return float(matches / len(predictions))


def precision(
    predictions: Field | Dataset, target: Field, *, average: str | None, positive=None, by: Field | None = None,
) -> float | Dataset:
    if by is not None:
        groups = metric_groups(predictions, target, by, "Precision", classification=(average, positive))
        scores = [precision(predicted, expected, average=average, positive=positive)
                  for predicted, expected in groups.pairs]
        return grouped_result(groups, scores, "precision")
    data = classification_data(predictions, target, average, positive)
    return classification_result(data, data.tp, data.tp + data.fp, average, positive, "precision")


def recall(
    predictions: Field | Dataset, target: Field, *, average: str | None, positive=None, by: Field | None = None,
) -> float | Dataset:
    if by is not None:
        groups = metric_groups(predictions, target, by, "Recall", classification=(average, positive))
        scores = [recall(predicted, expected, average=average, positive=positive)
                  for predicted, expected in groups.pairs]
        return grouped_result(groups, scores, "recall")
    data = classification_data(predictions, target, average, positive)
    return classification_result(data, data.tp, data.tp + data.fn, average, positive, "recall")


sensitivity = recall


def specificity(
    predictions: Field | Dataset, target: Field, *, average: str | None, positive=None, by: Field | None = None,
) -> float | Dataset:
    if by is not None:
        groups = metric_groups(predictions, target, by, "Specificity", classification=(average, positive))
        scores = [specificity(predicted, expected, average=average, positive=positive)
                  for predicted, expected in groups.pairs]
        return grouped_result(groups, scores, "specificity")
    data = classification_data(predictions, target, average, positive)
    return classification_result(data, data.tn, data.tn + data.fp, average, positive, "specificity")


def f1(
    predictions: Field | Dataset, target: Field, *, average: str | None, positive=None, by: Field | None = None,
) -> float | Dataset:
    if by is not None:
        groups = metric_groups(predictions, target, by, "F1", classification=(average, positive))
        scores = [f1(predicted, expected, average=average, positive=positive)
                  for predicted, expected in groups.pairs]
        return grouped_result(groups, scores, "f1")
    data = classification_data(predictions, target, average, positive)
    return classification_result(data, 2 * data.tp, 2 * data.tp + data.fp + data.fn, average, positive, "f1")


def mae(predictions: Field | Dataset, target: Field, *, by: Field | None = None) -> float | Dataset:
    if by is not None:
        groups = metric_groups(predictions, target, by, "MAE", numeric=True)
        scores = [mae(predicted, expected) for predicted, expected in groups.pairs]
        return grouped_result(groups, scores, "mae")
    residuals, _ = regression_data(predictions, target, "MAE")
    scale = np.abs(residuals).max()
    return float(scale * np.mean(np.abs(residuals / scale))) if scale else 0.0


def rmse(predictions: Field | Dataset, target: Field, *, by: Field | None = None) -> float | Dataset:
    if by is not None:
        groups = metric_groups(predictions, target, by, "RMSE", numeric=True)
        scores = [rmse(predicted, expected) for predicted, expected in groups.pairs]
        return grouped_result(groups, scores, "rmse")
    residuals, _ = regression_data(predictions, target, "RMSE")
    scale = np.abs(residuals).max()
    return float(scale * np.sqrt(np.mean((residuals / scale)**2))) if scale else 0.0


def r2(predictions: Field | Dataset, target: Field, *, by: Field | None = None) -> float | Dataset:
    if by is not None:
        groups = metric_groups(predictions, target, by, "R2", numeric=True)
        scores = [r2(predicted, expected) for predicted, expected in groups.pairs]
        return grouped_result(groups, scores, "r2")
    residuals, expected = regression_data(predictions, target, "R2")
    if len(expected) < 2:
        warnings.warn("R2 requires at least two samples; returning NaN", UserWarning, stacklevel=2)
        return float("nan")
    offsets = numeric_differences(expected, [expected[0]] * len(expected))
    target_scale = np.abs(offsets).max()
    error_scale = np.abs(residuals).max()
    if not target_scale:
        warnings.warn("R2 is undefined for a constant target; returning NaN or -inf", UserWarning, stacklevel=2)
        return float("nan") if not error_scale else float("-inf")
    if not error_scale:
        return 1.0
    centred = offsets / target_scale - np.mean(offsets / target_scale)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ratio = (error_scale / target_scale)**2 * np.sum((residuals / error_scale)**2) / np.sum(centred**2)
    return float(1 - ratio)
