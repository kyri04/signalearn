from signalearn.utility import *
from signalearn.preprocess import sample
from signalearn.classes import *
from signalearn.learning_utility import *
from sklearn.metrics import confusion_matrix
import numpy as np

def classify(
    points,
    y_attr,
    label,
    group=None,
    algorithm="rf",
    test_size=0.2,
    split_state=42,
    scale=False
):

    N = len(points)
    
    ys = np.array([getattr(point, y_attr) for point in points])
    
    labels = prepare_labels(points, label)
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

    model = get_classifier(algorithm)
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
    unique_cnts = {a: count_unique(points, a) for a in sorted(pt0_attrs)}
    test_meta = {a: np.array([getattr(points[i], a) for i in test_idx]) for a in pt0_attrs}

    res = Result(
        set_volume={
            "points": N,
            "classes": len(unique_labels),
            **unique_cnts
        },
        set_params={
            "label": label,
            "group": group,
            "algorithm": algorithm,
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
            "test_meta": test_meta,
            "model": model,
        }
    )
    return res

def regress(
    points,
    y_attr,
    target,
    group=None,
    algorithm="rf",
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

    model = get_regressor(algorithm)
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
    unique_cnts = {a: count_unique(points, a) for a in sorted(pt0_attrs)}
    test_meta = {a: np.array([getattr(points[i], a) for i in test_idx]) for a in pt0_attrs}

    res = Result(
        set_volume={
            "points": N,
            **unique_cnts
        },
        set_params={
            "target": target,
            "group": group,
            "algorithm": algorithm,
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
            "test_meta": test_meta,
            "model": model,
        }
    )
    return res

def shuffle_learn(
    points, 
    label, 
    y_attr,
    learn_func,
    group=None, 
    algorithm='rf', 
    test_size=0.2,
    shuffles=5,
    scale=False
):
    results = []
    for rs in range(shuffles):
        results.append(learn_func(
            points=points, 
            y_attr=y_attr,
            label = label, 
            group = group,
            algorithm = algorithm, 
            test_size = test_size,  
            split_state = rs,
            scale=scale))
        
    return results

def attr_curve(
    points, 
    y_attr,
    label, 
    by_attribute,
    learn_func,
    group=None, 
    algorithm='rf', 
    test_size=0.2,
    split_state=42,
    divisions=5,
    start_val=0,
    shuffles_per_split=None,
    scale=False
):

    rng = np.random.default_rng(split_state)

    values = np.array([getattr(p, by_attribute) for p in points])
    unique_vals = np.unique(values)
    rng.shuffle(unique_vals)
    n_unique = len(unique_vals)

    if n_unique < 1:
        return np.array([0], dtype=int), [np.nan]

    start_k = start_val
    counts = np.ceil(np.linspace(start_k, n_unique, divisions)).astype(int)
    counts = np.unique(np.clip(counts, 1, n_unique))

    results = []
    for k in counts:
        chosen = set(unique_vals[:k])
        subset = [p for p in points if getattr(p, by_attribute) in chosen]

        if (shuffles_per_split is not None) and (shuffles_per_split > 1):
            res_list = shuffle_learn(
                subset=subset,
                y_attr=y_attr,
                label=label,
                learn_func=learn_func,
                group=group,
                algorithm=algorithm,
                test_size=test_size,
                shuffles=shuffles_per_split,
                scale=scale
            )
            results.append(combine_results(res_list))
            
        else:
            res = learn_func(
                points=subset,
                y_attr=y_attr,
                label=label,
                group=group,
                algorithm=algorithm,
                test_size=test_size,
                split_state=split_state,
                scale=scale
            )
            results.append(res)

    return results

def data_curve(
    points, 
    label, 
    y_attr,
    learn_func,
    group=None, 
    algorithm='rf', 
    test_size=0.2,
    split_state=42, 
    divisions=5,
    start_fraction=0.05,
    shuffles_per_split=None,
    scale=False
):

    fractions = np.linspace(start_fraction, 1.0, divisions)
    results = []

    for frac in fractions:

        subset = sample(points, frac)
        if (shuffles_per_split is not None) and (shuffles_per_split > 1):
            res_list = shuffle_learn(
                points=subset,
                y_attr=y_attr,
                label=label,
                learn_func=learn_func,
                group=group,
                algorithm=algorithm,
                test_size=test_size,
                shuffles=shuffles_per_split,
                scale=scale
            )
            res = combine_results(res_list)
            results.append(res)
            
        else:
            res = learn_func(
                points=subset,
                y_attr=y_attr,
                label=label,
                group=group,
                algorithm=algorithm,
                test_size=test_size,
                split_state=split_state,
                scale=scale
            )
            results.append(res)

    return results