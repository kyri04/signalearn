from signalearn.utility import *
from signalearn.classes import *
from signalearn.learning_utility import *
from sklearn.metrics import confusion_matrix
import numpy as np

def classify(
    points,
    y_attr,
    target,
    group=None,
    model=RandomForestClassifier(),
    test_size=0.2,
    split_state=42,
    scale=False
):

    N = len(points)
    
    ys = np.array([getattr(point, y_attr) for point in points])
    
    labels = prepare_labels(points, target)
    groups = prepare_groups(points, group)

    labels_encoded, encoder = encode(labels)
    unique_labels = encoder.classes_
    unique_labels_encoded = np.unique(labels_encoded)

    train_idx, test_idx = get_single_split(N, labels_encoded, groups, test_size, split_state)

    if groups is not None:
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert train_groups.isdisjoint(test_groups)

    X_train_raw, X_test_raw = ys[train_idx], ys[test_idx]
    y_train, y_test = labels_encoded[train_idx], labels_encoded[test_idx]

    if(scale): X_train, X_test = standardize_train_test(X_train_raw, X_test_raw)
    else: X_train, X_test = X_train_raw, X_test_raw

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=unique_labels_encoded)

    n_classes = len(unique_labels_encoded)
    y_score = positive_class_scores(model, X_test, n_classes=n_classes)
    if n_classes == 2 and y_score.ndim > 1:
            y_score = y_score[:, 1]

    accuracy, mean_specificity, mean_sensitivity, mean_precision, mean_recall, mean_f1 = calculate_metrics(conf_matrix)

    feature_importances = get_feature_importances(model)

    pt0_attrs   = set(points[0].__dict__) - set(get_attributes(Series))
    test_meta = {a: np.array([getattr(points[i], a) for i in test_idx]) for a in pt0_attrs}

    res = Result(
        set_params={
            "target": target,
            "group": group,
            "model": model.__class__.__name__,
            "test_size": test_size,
            "split_state": split_state,
            "unique_labels": unique_labels,
            "mode": "normal"
        },
        set_results={
            "accuracy": accuracy,
            "specificity": mean_specificity,
            "sensitivity": mean_sensitivity,
            "precision": mean_precision,
            "recall": mean_recall,
            "f1": mean_f1
        },
        set_meta = {
            "conf_matrix": conf_matrix,
            "y_true": y_test,
            "y_score": y_score,
            "feature_importances": feature_importances,
            "test_index": test_idx,
            "test_meta": test_meta
        }
    )
    return res

def regress(
    points,
    y_attr,
    target,
    group=None,
    model=RandomForestRegressor(),
    test_size=0.2,
    split_state=42,
    scale=False
):
    N = len(points)

    ys = np.array([getattr(point, y_attr) for point in points])
    y = np.array([getattr(point, target) for point in points], dtype=float)
    groups = prepare_groups(points, group)

    train_idx, test_idx = get_single_split(N, None, groups, test_size, split_state)

    if groups is not None:
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert train_groups.isdisjoint(test_groups)

    X_train_raw, X_test_raw = ys[train_idx], ys[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if scale:
        X_train, X_test = standardize_train_test(X_train_raw, X_test_raw)
    else:
        X_train, X_test = X_train_raw, X_test_raw

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = np.mean((y_test - y_pred) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_test - y_pred)))
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    feature_importances = get_feature_importances(model)

    pt0_attrs   = set(points[0].__dict__) - set(get_attributes(Series))
    test_meta = {a: np.array([getattr(points[i], a) for i in test_idx]) for a in pt0_attrs}

    res = Result(
        set_params={
            "target": target,
            "group": group,
            "model": model.__class__.__name__,
            "test_size": test_size,
            "split_state": split_state,
            "mode": "normal"
        },
        set_results={
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2
        },
        set_meta={
            "y_true": y_test,
            "y_pred": y_pred,
            "residuals": y_test - y_pred,
            "feature_importances": feature_importances,
            "test_index": test_idx,
            "test_meta": test_meta
        }
    )
    return res
