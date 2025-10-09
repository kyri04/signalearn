from signalearn.utility import *
from signalearn.preprocess import sample
from signalearn.classes import ClassificationResult
from signalearn.learning_utility import *
from sklearn.metrics import confusion_matrix
import numpy as np

def classify(
    points,
    label,
    group=None,
    classifier="rf",
    test_size=0.2,
    split_state=42,
    agg_method='mean',
    agg_group=None,
    scale=False
):

    N = len(points)

    ys = np.array([point.y for point in points])
    
    labels = prepare_labels(points, label)
    groups = prepare_groups(points, group)
    agg_groups = prepare_groups(points, agg_group) if agg_group is not None else None

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

    model = get_classifier(classifier)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=unique_labels_encoded)

    n_classes = len(unique_labels_encoded)
    y_score = positive_class_scores(model, X_test, n_classes=n_classes)
    if n_classes == 2 and y_score.ndim > 1:
            y_score = y_score[:, 1]

    accuracy, mean_specificity, mean_sensitivity, mean_precision, mean_recall, mean_f1 = calculate_metrics(conf_matrix)

    group_results_ns = None
    if agg_groups is not None:
        groups_test = agg_groups[test_idx]
        group_eval = evaluate_group_level(
            y_true=y_test,
            y_score=y_score,
            groups=groups_test,
            unique_labels_encoded=unique_labels_encoded,
            agg=agg_method,
            threshold=0.5
        )
        group_results_ns = make_namespace(group_eval)

    res = ClassificationResult(
        set_params=make_namespace({
            "label": label,
            "group": group,
            "agg_group": agg_group if agg_group is not None else 'none',
            "classifier": classifier,
            "test_size": test_size,
            "split_state": split_state,
            "unique_labels": unique_labels,
            "group_agg": agg_method
        }),
        set_results=make_namespace({
            "accuracy": accuracy,
            "specificity": mean_specificity,
            "sensitivity": mean_sensitivity,
            "precision": mean_precision,
            "recall": mean_recall,
            "f1": mean_f1,
            "conf_matrix": conf_matrix,
            "y_true": y_test,
            "y_score": y_score,
        }),
        set_group_results=group_results_ns
    )

    return res

def shuffle_classify(
    points, 
    label, 
    group=None, 
    agg_group=None,
    classifier='rf', 
    test_size=0.2,
    shuffles=5,
    agg_method='mean',
    scale=False
):
    results = []
    for rs in range(shuffles):
        results.append(classify(
            points=points, 
            label = label, 
            group = group, 
            agg_group = agg_group,
            classifier = classifier, 
            test_size = test_size,  
            split_state = rs,
            agg_method = agg_method,
            scale=scale))
        
    return results

def learning_curve(
    points, 
    label, 
    by_attribute,
    group=None, 
    agg_group=None,
    classifier='rf', 
    test_size=0.2,
    split_state=42,
    divisions=5,
    start_val=4,
    shuffles_per_split=None,
    agg_method='mean',
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

    scores = []
    group_scores = []
    for k in counts:
        chosen = set(unique_vals[:k])
        subset = [p for p in points if getattr(p, by_attribute) in chosen]

        if group is not None:
            subset_groups = {getattr(p, group) for p in subset}
            if len(subset_groups) < 2:
                scores.append(np.nan)
                continue

        if (shuffles_per_split is not None) and (shuffles_per_split > 1):
            res_list = shuffle_classify(
                subset=subset,
                label=label,
                group=group,
                agg_group=agg_group,
                classifier=classifier,
                test_size=test_size,
                shuffles=shuffles_per_split,
                agg_method=agg_method,
                scale=scale
            )
            f1_scores = [res.results.f1 for res in res_list if not np.isnan(res.results.f1)]
            mean_f1 = np.mean(f1_scores) if len(f1_scores) > 0 else np.nan
            scores.append(mean_f1)

            group_f1_scores = [res.group_results.f1 for res in res_list if not np.isnan(res.group_results.f1)]
            mean_group_f1 = np.mean(group_f1_scores) if len(group_f1_scores) > 0 else np.nan
            group_scores.append(mean_group_f1)
            
        else:
            res = classify(
                points=subset,
                label=label,
                group=group,
                agg_group=agg_group,
                classifier=classifier,
                test_size=test_size,
                split_state=split_state,
                agg_method=agg_method,
                scale=scale
            )
            scores.append(res.results.f1)
            group_scores.append(res.group_results.f1)

    return counts, scores, group_scores

def score_curve(
    points, 
    label, 
    group=None, 
    agg_group=None,
    classifier='rf', 
    test_size=0.2,
    split_state=42, 
    divisions=5,
    start_fraction=0.05,
    shuffles_per_split=None,
    agg_method='mean',
    scale=False
):

    fractions = np.linspace(start_fraction, 1.0, divisions)
    scores = []
    group_scores = []
    for frac in fractions:

        subset = sample(points, frac)
        if (shuffles_per_split is not None) and (shuffles_per_split > 1):
            res_list = shuffle_classify(
                points=subset,
                label=label,
                group=group,
                agg_group=agg_group,
                classifier=classifier,
                test_size=test_size,
                shuffles=shuffles_per_split,
                agg_method=agg_method,
                scale=scale
            )
            f1_scores = [res.results.f1 for res in res_list if not np.isnan(res.results.f1)]
            mean_f1 = np.mean(f1_scores) if len(f1_scores) > 0 else np.nan
            scores.append(mean_f1)

            group_f1_scores = [res.group_results.f1 for res in res_list if not np.isnan(res.group_results.f1)]
            mean_group_f1 = np.mean(group_f1_scores) if len(group_f1_scores) > 0 else np.nan
            group_scores.append(mean_group_f1)
            
        else:
            res = classify(
                points=subset,
                label=label,
                group=group,
                agg_group=agg_group,
                classifier=classifier,
                test_size=test_size,
                split_state=split_state,
                agg_method=agg_method,
                scale=scale
            )
            scores.append(res.results.f1)
            group_scores.append(res.group_results.f1)

    return fractions, scores, group_scores