from signalearn.utility import *
from signalearn.preprocess import sample
from signalearn.classes import ClassificationResult
from signalearn.learning_utility import *
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from joblib import Parallel, delayed
import numpy as np

def classify(points, label, group=None, classifier="rf",
             test_size=0.2, split_state=None, agg_method='mean', agg_group=None):
    """
    Perform one train-test split classification. If split_state is an integer,
    we use it as random_state; if None, use default behavior.
    """
    N = len(points)
    # Extract target values and prepare label encoding only once per call
    ys = np.ascontiguousarray([p.y for p in points], dtype=np.float32)
    labels = prepare_labels(points, label)
    groups = prepare_groups(points, group)
    agg_groups = prepare_groups(points, agg_group) if agg_group is not None else None

    # Encode labels to integers
    labels_encoded, encoder = encode(labels)
    unique_labels = encoder.classes_
    unique_labels_encoded = np.unique(labels_encoded)

    # Determine train/test split indices
    if split_state is not None:
        # Use the specified random seed for reproducibility
        train_idx, test_idx = get_single_split(N, labels_encoded, groups,
                                               test_size=test_size, random_state=split_state)
    else:
        # split_state None: use default split (e.g., random_state=42)
        train_idx, test_idx = get_single_split(N, labels_encoded, groups,
                                               test_size=test_size, random_state=42)

    # Ensure group separation if groups provided
    if groups is not None:
        assert set(groups[train_idx]).isdisjoint(groups[test_idx])

    # Split features and target
    X_train_raw, X_test_raw = ys[train_idx], ys[test_idx]
    y_train, y_test = labels_encoded[train_idx], labels_encoded[test_idx]

    # Standardize features (fit scaler on train only)
    X_train, X_test = standardize_train_test(X_train_raw, X_test_raw)

    # Fit classifier and predict
    model = get_classifier(classifier)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=unique_labels_encoded)

    # Compute raw scores (for ROC, etc.)
    n_classes = len(unique_labels_encoded)
    y_score = positive_class_scores(model, X_test, n_classes=n_classes)
    if n_classes == 2 and y_score.ndim > 1:
        # For binary, take probability of positive class
        y_score = y_score[:, 1]

    # Compute metrics from confusion matrix
    accuracy, mean_specificity, mean_sensitivity, mean_precision, mean_recall, mean_f1 \
        = calculate_metrics(conf_matrix)

    # Group-level evaluation if requested
    group_results_ns = None
    if agg_groups is not None:
        groups_test = agg_groups[test_idx]
        grp_eval = evaluate_group_level(
            y_true=y_test,
            y_score=y_score,
            groups=groups_test,
            unique_labels_encoded=unique_labels_encoded,
            agg=agg_method,
            threshold=0.5
        )
        group_results_ns = make_namespace(grp_eval)

    # Package into ClassificationResult (preserving structure)
    params = {
        "label": label,
        "group": group,
        "agg_group": agg_group if agg_group is not None else 'none',
        "classifier": classifier,
        "test_size": test_size,
        "split_state": split_state,
        "unique_labels": unique_labels,
        "group_agg": agg_method
    }
    results = {
        "accuracy": accuracy,
        "specificity": mean_specificity,
        "sensitivity": mean_sensitivity,
        "precision": mean_precision,
        "recall": mean_recall,
        "f1": mean_f1,
        "conf_matrix": conf_matrix,
        "y_true": y_test,
        "y_score": y_score
    }
    return ClassificationResult(
        set_params=make_namespace(params),
        set_results=make_namespace(results),
        set_group_results=group_results_ns
    )

def shuffle_classify(points, label, group=None, agg_group=None,
                     classifier='rf', test_size=0.2, shuffles=5, agg_method='mean'):
    """
    Perform multiple classifications with different random splits.  Uses StratifiedShuffleSplit
    or GroupShuffleSplit to generate all splits, and runs fits in parallel.
    Returns a list of ClassificationResult objects (one per split).
    """
    N = len(points)
    ys = np.ascontiguousarray([p.y for p in points], dtype=np.float32)
    labels = prepare_labels(points, label)
    groups = prepare_groups(points, group)
    labels_encoded, _ = encode(labels)  # we only need encoded labels for splitting

    # Select cross-validator: group-aware or stratified
    if groups is not None:
        cv = GroupShuffleSplit(n_splits=shuffles, test_size=test_size, random_state=42)
        split_iter = cv.split(np.arange(N), labels_encoded, groups)
    else:
        cv = StratifiedShuffleSplit(n_splits=shuffles, test_size=test_size, random_state=42)
        split_iter = cv.split(np.arange(N), labels_encoded, labels_encoded)

    # Run classification on each split (in parallel)
    def _run_split(train_idx, test_idx):
        # Similar to classify(), but we already have indices
        X_train_raw, X_test_raw = ys[train_idx], ys[test_idx]
        y_train, y_test = labels_encoded[train_idx], labels_encoded[test_idx]
        X_train, X_test = standardize_train_test(X_train_raw, X_test_raw)
        model = get_classifier(classifier)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred, labels=np.unique(labels_encoded))
        n_classes = len(np.unique(labels_encoded))
        y_score = positive_class_scores(model, X_test, n_classes=n_classes)
        if n_classes == 2 and y_score.ndim > 1:
            y_score = y_score[:, 1]
        accuracy, mean_specificity, mean_sensitivity, mean_precision, mean_recall, mean_f1 \
            = calculate_metrics(conf_matrix)
        # Group-level metrics
        group_results_ns = None
        if agg_group is not None:
            agg_groups = prepare_groups(points, agg_group)
            groups_test = agg_groups[test_idx]
            grp_eval = evaluate_group_level(
                y_true=y_test,
                y_score=y_score,
                groups=groups_test,
                unique_labels_encoded=np.unique(labels_encoded),
                agg=agg_method,
                threshold=0.5
            )
            group_results_ns = make_namespace(grp_eval)
        params = {
            "label": label, "group": group, "agg_group": agg_group if agg_group else 'none',
            "classifier": classifier, "test_size": test_size,
            "split_state": None,  # no fixed seed here
            "unique_labels": np.unique(labels), "group_agg": agg_method
        }
        results = {
            "accuracy": accuracy, "specificity": mean_specificity, "sensitivity": mean_sensitivity,
            "precision": mean_precision, "recall": mean_recall, "f1": mean_f1,
            "conf_matrix": conf_matrix, "y_true": y_test, "y_score": y_score
        }
        return ClassificationResult(
            set_params=make_namespace(params),
            set_results=make_namespace(results),
            set_group_results=group_results_ns
        )

    # Launch parallel execution for each train/test split
    splits = list(split_iter)
    results = [_run_split(tr, te) for tr, te in splits]

    # Better performance on CPU, do not use on GPU:
    # results = Parallel(n_jobs=-1)(
    #     delayed(_run_split)(train_idx, test_idx) for train_idx, test_idx in splits
    # )
    return results

def learning_curve(points, label, by_attribute, group=None, agg_group=None,
                   classifier='rf', test_size=0.2, split_state=42,
                   divisions=5, start_val=4, shuffles_per_split=None, agg_method='mean'):
    """
    Compute a “learning curve” by progressively adding unique values of `by_attribute`.
    At each step, run (shuffled) classification to get performance (F1) as data size grows.
    """
    rng = np.random.default_rng(split_state)
    # Shuffle the unique attribute values
    values = np.array([getattr(p, by_attribute) for p in points])
    unique_vals = np.unique(values)
    rng.shuffle(unique_vals)
    n_unique = len(unique_vals)
    if n_unique == 0:
        return np.array([0], dtype=int), [np.nan], [np.nan]

    # Determine how many values to include at each division
    start_k = start_val
    counts = np.ceil(np.linspace(start_k, n_unique, divisions)).astype(int)
    counts = np.unique(np.clip(counts, 1, n_unique))

    scores = []
    group_scores = []
    for k in counts:
        chosen_vals = set(unique_vals[:k])
        subset = [p for p in points if getattr(p, by_attribute) in chosen_vals]

        # If grouping is used but not enough groups, skip
        if group is not None:
            subset_groups = {getattr(p, group) for p in subset}
            if len(subset_groups) < 2:
                scores.append(np.nan)
                group_scores.append(np.nan)
                continue

        if shuffles_per_split and shuffles_per_split > 1:
            # Perform multiple shuffles and average F1 scores
            res_list = shuffle_classify(
                points=subset, label=label, group=group, agg_group=agg_group,
                classifier=classifier, test_size=test_size,
                shuffles=shuffles_per_split, agg_method=agg_method
            )
            f1_list = [res.results.f1 for res in res_list if res.results.f1 is not None]
            scores.append(np.mean(f1_list) if f1_list else np.nan)

            grp_list = [res.group_results.f1 for res in res_list
                        if res.group_results and res.group_results.f1 is not None]
            group_scores.append(np.mean(grp_list) if grp_list else np.nan)
        else:
            # Single split classification
            res = classify(
                points=subset, label=label, group=group, agg_group=agg_group,
                classifier=classifier, test_size=test_size,
                split_state=split_state, agg_method=agg_method
            )
            scores.append(res.results.f1)
            group_scores.append(res.group_results.f1 if res.group_results else np.nan)

    return counts, scores, group_scores

def score_curve(points, label, group=None, agg_group=None,
                classifier='rf', test_size=0.2, split_state=42,
                divisions=5, start_fraction=0.05, shuffles_per_split=None, agg_method='mean'):
    """
    Compute performance vs. fraction of data used.  At each fraction, randomly sample
    that portion of points, then classify (optionally averaging over shuffles).
    """
    fractions = np.linspace(start_fraction, 1.0, divisions)
    scores = []
    group_scores = []
    for frac in fractions:
        subset = sample(points, frac)  # random subset of given fraction

        if shuffles_per_split and shuffles_per_split > 1:
            res_list = shuffle_classify(
                points=subset, label=label, group=group, agg_group=agg_group,
                classifier=classifier, test_size=test_size,
                shuffles=shuffles_per_split, agg_method=agg_method
            )
            f1_list = [res.results.f1 for res in res_list if res.results.f1 is not None]
            scores.append(np.mean(f1_list) if f1_list else np.nan)

            grp_list = [res.group_results.f1 for res in res_list
                        if res.group_results and res.group_results.f1 is not None]
            group_scores.append(np.mean(grp_list) if grp_list else np.nan)
        else:
            res = classify(
                points=subset, label=label, group=group, agg_group=agg_group,
                classifier=classifier, test_size=test_size,
                split_state=split_state, agg_method=agg_method
            )
            scores.append(res.results.f1)
            group_scores.append(res.group_results.f1 if res.group_results else np.nan)

    return fractions, scores, group_scores
