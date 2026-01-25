import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from signalearn.classes import Sample
from signalearn.utility import as_fields

def clf_kwargs(y_true):
    uniq = np.unique(np.asarray(y_true, dtype=object))
    if uniq.size == 2:
        return {"average": "binary", "pos_label": uniq[1], "zero_division": 0}
    return {"average": "macro", "zero_division": 0}

def prepare_labels(label):
    if isinstance(label, (list, tuple)):
        n = len(label[0].fields)
        labels = []
        for i in range(n):
            parts = [str(fc.fields[i].values) for fc in label]
            labels.append("_".join(parts))
        return np.array(labels, dtype=str)
    return np.array([f.values for f in label.fields], dtype=str)

def prepare_groups(group):
    return np.array([f.values for f in group.fields], dtype=str) if group is not None else None

def build_feature_matrix(y_attr):
    if isinstance(y_attr, (list, tuple)):
        attrs = list(y_attr)
    else:
        attrs = [y_attr]

    feature_blocks = []
    matrices = []
    start = 0
    for attr in attrs:
        first = np.asarray(attr.fields[0].values, dtype=float).ravel()
        length = first.size
        block = np.empty((len(attr.fields), length), dtype=float)
        block[0] = first
        for idx, field in enumerate(attr.fields[1:], start=1):
            values = np.asarray(field.values, dtype=float).ravel()
            block[idx] = values

        matrices.append(block)
        feature_blocks.append({"attr": attr.name, "start": start, "stop": start + length, "length": length})
        start += length

    X = np.hstack(matrices) if len(matrices) > 1 else matrices[0]
    return X, tuple(a.name for a in attrs), tuple(feature_blocks)

def first(x):
    return x[0] if isinstance(x, (list, tuple)) else x

def get(dataset, x):
    if isinstance(x, (list, tuple)):
        return [getattr(dataset, fc.name) for fc in x]
    return getattr(dataset, x.name)

def values(x):
    if isinstance(x, (list, tuple)):
        n = len(x[0].fields)
        return [tuple(fc.fields[i].values for fc in x) for i in range(n)]
    return [f.values for f in x.fields]

def eval_once(train, test, x, target, model, fit, group=None, meta=None, extra=None):
    x_train = get(train, x)
    x_test = get(test, x)
    y_train = get(train, target)
    y_test = get(test, target)

    model = fit(x_train, y_train, model)

    X, _, _ = build_feature_matrix(as_fields(x_test))
    y_pred = model.predict(X)

    y_score = None
    score_fn = getattr(model, "predict_proba", None) or getattr(model, "decision_function", None)
    if score_fn is not None:
        y_score = score_fn(X)
        if getattr(y_score, "ndim", 0) == 2 and y_score.shape[1] == 2:
            y_score = y_score[:, 1]

    y_true = values(y_test)

    id_name = "id"
    id_vals = values(test.id)

    group_name = None
    group_vals = None
    if group is not None:
        group_name = group.name
        group_vals = values(get(test, group))

    out = []
    base = meta or {}
    extras = extra or {}
    for i in range(len(test.samples)):
        row = dict(base)
        row.update(extras)
        row["y_true"] = y_true[i]
        row["y_pred"] = y_pred[i]
        if y_score is not None:
            row["y_score"] = y_score[i]
        row[id_name] = id_vals[i]
        if group_name is not None:
            row[group_name] = group_vals[i]
        out.append(Sample(row))

    return out

def get_single_split(N, y, groups, test_size=0.2, random_state=42):
    if groups is not None:
        gss = GroupShuffleSplit(test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(np.arange(N), y, groups))
    else:
        train_idx, test_idx = train_test_split(
            np.arange(N), test_size=test_size, stratify=y, random_state=random_state
        )
    return train_idx, test_idx
