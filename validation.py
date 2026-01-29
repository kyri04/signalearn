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
from sklearn.base import clone

from signalearn.utility import as_fields
from signalearn.learning_utility import build_feature_matrix, prepare_labels, clf_kwargs, first, eval_once, values, get
from signalearn.learning import split
from signalearn.classes import Dataset, Sample

def predict(x, target, model):
    dataset = first(x)._dataset
    if target._dataset is not dataset:
        target = get(dataset, target)
    X, _, _ = build_feature_matrix(as_fields(x))
    y_pred = model.predict(X)
    classes = getattr(model, "_signalearn_classes", None)
    if classes is not None:
        y_pred = np.asarray(y_pred, dtype=int)
        y_pred = classes[y_pred]

    y_score = None
    score_fn = getattr(model, "predict_proba", None) or getattr(model, "decision_function", None)
    if score_fn is not None:
        y_score = np.asarray(score_fn(X))
        if y_score.ndim == 2 and y_score.shape[1] == 2:
            y_score = y_score[:, 1]

    y_true = values(target)
    id_vals = values(dataset.id)

    out = []
    for i in range(len(dataset.samples)):
        row = {"y_true": y_true[i], "y_pred": y_pred[i], "id": id_vals[i]}
        if y_score is not None:
            row["y_score"] = y_score[i]
        out.append(Sample(row))

    return Dataset(out)

def accuracy(predictions):
    y_true = prepare_labels(predictions.y_true)
    y_pred = np.asarray(values(predictions.y_pred), dtype=object)
    return float(accuracy_score(y_true, y_pred))

def precision(predictions):
    y_true = prepare_labels(predictions.y_true)
    y_pred = np.asarray(values(predictions.y_pred), dtype=object)
    return float(precision_score(y_true, y_pred, **clf_kwargs(y_true)))

def recall(predictions):
    y_true = prepare_labels(predictions.y_true)
    y_pred = np.asarray(values(predictions.y_pred), dtype=object)
    return float(recall_score(y_true, y_pred, **clf_kwargs(y_true)))

def f1(predictions):
    y_true = prepare_labels(predictions.y_true)
    y_pred = np.asarray(values(predictions.y_pred), dtype=object)
    return float(f1_score(y_true, y_pred, **clf_kwargs(y_true)))

def sensitivity(predictions):
    return recall(predictions)

def specificity(predictions):
    y_true = prepare_labels(predictions.y_true)
    y_pred = np.asarray(values(predictions.y_pred), dtype=object)
    labels = np.unique(np.concatenate([np.asarray(y_true, dtype=object), y_pred], axis=0))
    if labels.size != 2:
        return None
    neg = labels[0]
    return float(recall_score(y_true, y_pred, average="binary", pos_label=neg, zero_division=0))

def auc(predictions):
    if not hasattr(predictions, "y_score"):
        return None
    y_true = prepare_labels(predictions.y_true)
    y_score = np.asarray(values(predictions.y_score), dtype=float)
    try:
        return float(roc_auc_score(y_true, y_score, multi_class="ovr"))
    except Exception:
        return None

def mse(predictions):
    y_true = np.asarray(values(predictions.y_true), dtype=float)
    y_pred = np.asarray(values(predictions.y_pred), dtype=float)
    return float(mean_squared_error(y_true, y_pred))

def rmse(predictions):
    return float(np.sqrt(mse(predictions)))

def mae(predictions):
    y_true = np.asarray(values(predictions.y_true), dtype=float)
    y_pred = np.asarray(values(predictions.y_pred), dtype=float)
    return float(mean_absolute_error(y_true, y_pred))

def r2(predictions):
    y_true = np.asarray(values(predictions.y_true), dtype=float)
    y_pred = np.asarray(values(predictions.y_pred), dtype=float)
    return float(r2_score(y_true, y_pred))

def crossval(
    x,
    target,
    model,
    fit,
    group=None,
    test_size=0.2,
    repeats=5,
    seed=42,
):
    dataset = first(x)._dataset

    out = []
    for fold in range(int(repeats)):
        fold_seed = seed + fold
        sp = split(dataset, group=group, test_size=test_size, seed=fold_seed)
        out.extend(
            eval_once(
                train=sp.train,
                test=sp.test,
                x=x,
                target=target,
                model=clone(model),
                fit=fit,
                group=group,
                meta={"run": fold, "fold": fold, "seed": fold_seed},
            )
        )

    return Dataset(out)

def tune(
    x,
    target,
    model,
    fit,
    params,
    group=None,
    test_size=0.2,
    repeats=5,
    seed=42,
):
    dataset = first(x)._dataset

    out = []
    for trial, p in enumerate(params):
        for fold in range(int(repeats)):
            fold_seed = seed + fold
            sp = split(dataset, group=group, test_size=test_size, seed=fold_seed)
            m = clone(model)
            m.set_params(**p)
            out.extend(
                eval_once(
                    train=sp.train,
                    test=sp.test,
                    x=x,
                    target=target,
                    model=m,
                    fit=fit,
                    group=group,
                    meta={"run": trial * int(repeats) + fold, "trial": trial, "fold": fold, "seed": fold_seed},
                    extra=p,
                )
            )

    return Dataset(out)

def learncurve(
    x,
    target,
    model,
    fit,
    sizes,
    group=None,
    test_size=0.2,
    repeats=5,
    seed=42,
):
    dataset = first(x)._dataset

    out = []
    run = 0
    for fold in range(int(repeats)):
        fold_seed = seed + fold
        sp = split(dataset, group=group, test_size=test_size, seed=fold_seed)
        train = sp.train
        test = sp.test

        if group is not None:
            gname = group.name
            gids = np.array([s.fields[gname].values for s in train.samples], dtype=object)
            uniq = np.unique(gids)
            rng = np.random.default_rng(fold_seed)
            rng.shuffle(uniq)

            def subset(n):
                sel = set(uniq[: min(int(n), uniq.size)].tolist())
                return Dataset([s for s in train.samples if s.fields[gname].values in sel])

        else:
            rng = np.random.default_rng(fold_seed)
            order = rng.permutation(len(train.samples))

            def subset(n):
                m = min(int(n), len(train.samples))
                return Dataset([train.samples[i] for i in order[:m]])

        for train_size in sizes:
            train_sub = subset(train_size)
            out.extend(
                eval_once(
                    train=train_sub,
                    test=test,
                    x=x,
                    target=target,
                    model=clone(model),
                    fit=fit,
                    group=group,
                    meta={
                        "run": run,
                        "fold": fold,
                        "seed": fold_seed,
                        "train_size": int(train_size),
                    },
                )
            )
            run += 1

    return Dataset(out)
