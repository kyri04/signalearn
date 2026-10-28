from collections.abc import Sequence
from typing import Any

from ._internal.adapters import get_adapter
from ._internal.alignment import model_inputs, output_dataset, selected_fields
from ._internal.learning import deferred_prediction, validate_fit_parameters
from .data import Dataset, Field

__all__ = ["fit", "predict"]


def fit(
    x: Field | Sequence[Field] | Dataset, model: Any,
    target: Field | None = None, **fit_params: Any,
) -> Any:
    adapter = get_adapter(model, "fit")
    validate_fit_parameters(fit_params)
    fields, names, ids = selected_fields(x)
    matrix, labels, parameters = model_inputs(fields, target, fit_params, ids).compute()
    return adapter.fit_model(model, matrix, labels, parameters, fields, names, target)


def predict(
    x: Field | Sequence[Field] | Dataset, model: Any,
    *, output: str | None = None,
    **predict_params: Any,
) -> Dataset:
    adapter = get_adapter(model, "predict")
    fields, names, ids = selected_fields(x)
    fields, names = adapter.fitted_inputs(model, fields, names)
    schema = adapter.prediction_schema(model, output)
    inputs = model_inputs(fields, None, predict_params, ids, snapshot=True)
    result = deferred_prediction(adapter, model, inputs, len(ids), schema, output)
    return output_dataset(result, schema, ids)
