import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from signalearn.utility import as_fields
from signalearn.learning_utility import build_feature_matrix, prepare_labels, _clf_kwargs

def predict(x, model):
    X, _, _ = build_feature_matrix(as_fields(x))
    y_pred = model.predict(X)

    y_score = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if getattr(proba, "ndim", 0) == 2 and proba.shape[1] == 2:
            y_score = proba[:, 1]
        else:
            y_score = proba
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if getattr(scores, "ndim", 0) == 2 and scores.shape[1] == 2:
            y_score = scores[:, 1]
        else:
            y_score = scores

    return {"y_pred": y_pred, "y_score": y_score}

def accuracy(target, pred):
    y_true = prepare_labels(target)
    return float(accuracy_score(y_true, pred["y_pred"]))

def precision(target, pred):
    y_true = prepare_labels(target)
    return float(precision_score(y_true, pred["y_pred"], **_clf_kwargs(y_true)))

def recall(target, pred):
    y_true = prepare_labels(target)
    return float(recall_score(y_true, pred["y_pred"], **_clf_kwargs(y_true)))

def f1(target, pred):
    y_true = prepare_labels(target)
    return float(f1_score(y_true, pred["y_pred"], **_clf_kwargs(y_true)))

def auc(target, pred):
    y_true = prepare_labels(target)
    y_score = pred.get("y_score")
    return float(roc_auc_score(y_true, y_score, multi_class="ovr"))

def mse(target, pred):
    y_true = np.array([f.values for f in target.fields], dtype=float)
    y_pred = np.asarray(pred["y_pred"], dtype=float)
    return float(mean_squared_error(y_true, y_pred))

def rmse(target, pred):
    return float(np.sqrt(mse(target, pred)))

def mae(target, pred):
    y_true = np.array([f.values for f in target.fields], dtype=float)
    y_pred = np.asarray(pred["y_pred"], dtype=float)
    return float(mean_absolute_error(y_true, y_pred))

def r2(target, pred):
    y_true = np.array([f.values for f in target.fields], dtype=float)
    y_pred = np.asarray(pred["y_pred"], dtype=float)
    return float(r2_score(y_true, y_pred))
