from signalearn.utility import *
from signalearn.preprocess import sample
from signalearn.classes import *
from signalearn.learning_utility import *
from sklearn.metrics import confusion_matrix
import numpy as np

def classify(
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
            "target": target,
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

def ordinal_classify(
    points,
    y_attr,
    target,
    group=None,
    algorithm="rf",
    test_size=0.2,
    split_state=42,
    scale=False,
    cutoff=0.5
):
    N = len(points)

    X = np.asarray([getattr(p, y_attr) for p in points], dtype=float)

    y_all = np.array([int(getattr(p, target, None)) for p in points], dtype=object)
    mask = np.array([v is not None for v in y_all], dtype=bool)
    if not mask.all():
        X = X[mask]
        points = [p for i, p in enumerate(points) if mask[i]]
        y_all = y_all[mask]
        N = len(points)
    y_all = y_all.astype(int)

    classes = np.array(sorted(np.unique(y_all)))
    groups_arr = prepare_groups(points, group)

    train_idx, test_idx = get_single_split(N, y_all, groups_arr, test_size, split_state)
    if groups_arr is not None:
        tr_g, te_g = set(groups_arr[train_idx]), set(groups_arr[test_idx])
        assert tr_g.isdisjoint(te_g)

    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train_all, y_test_all = y_all[train_idx], y_all[test_idx]

    if scale:
        X_train, X_test = standardize_train_test(X_train_raw, X_test_raw)
    else:
        X_train, X_test = X_train_raw, X_test_raw

    thresholds = classes[1:]
    K = thresholds.size

    P_ge = np.zeros((X_test.shape[0], K), dtype=float)
    models = []
    imps = []

    for k_idx, thr in enumerate(thresholds):
        y_train_bin = (y_train_all >= thr).astype(int)

        if len(np.unique(y_train_bin)) < 2:
            const = float(y_train_bin.mean())
            P_ge[:, k_idx] = const
            models.append(None)
            imps.append(None)
            continue

        clf = get_classifier(algorithm)
        clf.fit(X_train, y_train_bin)

        n_classes = 2
        p = positive_class_scores(clf, X_test, n_classes=n_classes)
        if p.ndim > 1:
            p = p[:, 1]
        P_ge[:, k_idx] = p
        models.append(clf)

        fi = get_feature_importances(clf)
        imps.append(fi if fi is not None else None)

    probs = np.zeros((X_test.shape[0], classes.size), dtype=float)
    probs[:, 0] = 1.0 - P_ge[:, 0]
    if K > 1:
        probs[:, 1:-1] = P_ge[:, :-1] - P_ge[:, 1:]
    probs[:, -1] = P_ge[:, -1]

    probs = np.clip(probs, 0.0, 1.0)
    s = probs.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    probs /= s

    passes = (P_ge >= float(cutoff)).sum(axis=1)
    y_pred = classes[passes]
    y_true = y_test_all

    conf = confusion_matrix(y_true, y_pred, labels=classes)
    acc, spec, sens, prec, rec, f1 = calculate_metrics(conf)

    fi_stack = [np.asarray(v) for v in imps if v is not None]
    feat_imp = np.mean(fi_stack, axis=0) if fi_stack and len({v.shape for v in fi_stack}) == 1 else None

    pt0_attrs = set(points[train_idx[0]].__dict__) - set(get_attributes(Series))
    unique_cnts = {a: count_unique(points, a) for a in sorted(pt0_attrs)}
    test_meta = {a: np.array([getattr(points[i], a) for i in test_idx]) for a in pt0_attrs}

    return Result(
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
            "mode": "ordinal",
            "ordinal_thresholds": thresholds.tolist(),
            "cutoff": float(cutoff),
            "unique_labels": classes
        },
        set_results={
            "accuracy": acc,
            "specificity": spec,
            "sensitivity": sens,
            "precision": prec,
            "recall": rec,
            "f1": f1
        },
        set_meta={
            "conf_matrix": conf,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_score": probs,
            "threshold_scores": P_ge,
            "feature_importances": feat_imp,
            "test_index": test_idx,
            "test_meta": test_meta,
            "models": models
        }
    )

def shuffle_learn(
    points, 
    target, 
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
            target = target, 
            group = group,
            algorithm = algorithm, 
            test_size = test_size,  
            split_state = rs,
            scale=scale))
        
    return results

def attr_curve(
    points, 
    y_attr,
    target, 
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
                target=target,
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
                target=target,
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
    target, 
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
                target=target,
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
                target=target,
                group=group,
                algorithm=algorithm,
                test_size=test_size,
                split_state=split_state,
                scale=scale
            )
            results.append(res)

    return results