from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from collections import Counter
import numpy as np

from signalearn.classes import Result
from signalearn.utility import *

def reduce(y, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(y)

def encode(labels):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return np.array(encoded_labels), encoder

def prepare_labels(points, label):
    if isinstance(label, str):
        labels = np.array([getattr(point, label) for point in points], dtype=str)
    elif isinstance(label, list):
        labels = np.array(
            ["_".join(str(getattr(point, attr)) for attr in label) for point in points],
            dtype=str,
        )

    return labels

def prepare_groups(points, group):
    return (
        np.array([getattr(point, group) for point in points], dtype=str)
        if group is not None
        else None
    )

def build_name(point_type, label, group, classifier, attr_same, val_same):
    suffix = f"-{attr_same}={val_same}" if attr_same is not None and val_same is not None else ""
    return f"{point_type}-{label}{('-' + group) if group is not None else ''}-{classifier}{suffix}"

def get_feature_importances(model):
    feature_importances = None
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        feature_importances = np.abs(model.coef_).flatten()

    return feature_importances

def get_single_split(N, y, groups, test_size=0.2, random_state=42):
    if groups is not None:
        gss = GroupShuffleSplit(test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(np.arange(N), y, groups))
    else:
        train_idx, test_idx = train_test_split(
            np.arange(N), test_size=test_size, stratify=y, random_state=random_state
        )
    return train_idx, test_idx

def scale(y):
    scaler = StandardScaler()
    return np.array(scaler.fit_transform(y))

def standardize_train_test(X_train_raw, X_test_raw):
    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, X_test

def get_classifier(method):
    classifiers = {
        "dt": DecisionTreeClassifier(random_state=42),
        "rf": RandomForestClassifier(random_state=42, n_estimators=300, max_depth=10, class_weight='balanced'),
        "svm": SVC(random_state=42, probability=True),
        "lr": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        "knn": KNeighborsClassifier(),
        "gb": GradientBoostingClassifier(random_state=42),
    }
    return classifiers[method]

def calculate_metrics(conf_matrix, average="binary", pos_index=1):
    C = conf_matrix.shape[0]
    total_samples = conf_matrix.sum()
    accuracy = np.trace(conf_matrix) / total_samples if total_samples > 0 else None

    tps = np.diag(conf_matrix)
    fns = conf_matrix.sum(axis=1) - tps
    fps = conf_matrix.sum(axis=0) - tps
    tns = total_samples - (tps + fns + fps)

    with np.errstate(divide='ignore', invalid='ignore'):
        sens = np.where((tps + fns) > 0, tps / (tps + fns), np.nan)
        spec = np.where((tns + fps) > 0, tns / (tns + fps), np.nan)
        prec = np.where((tps + fps) > 0, tps / (tps + fps), np.nan)
        f1 = np.where(
            (prec + sens) > 0,
            2 * (prec * sens) / (prec + sens),
            np.nan
        )

    if average == "binary":
        i = pos_index
        mean_sensitivity = _nan_to_none(sens[i])
        mean_specificity = _nan_to_none(spec[i])
        mean_precision  = _nan_to_none(prec[i])
        mean_recall     = _nan_to_none(sens[i])
        mean_f1         = _nan_to_none(f1[i])

    else:
        supports = conf_matrix.sum(axis=1)
        if average == "weighted":
            weights = np.where(supports > 0, supports / supports.sum(), 0.0)
        elif average == "macro":
            weights = np.ones(C) / C
        else:
            raise ValueError("average must be 'macro', 'weighted', or 'binary'.")

        def wavg(x):
            x = np.where(np.isnan(x), 0.0, x)
            return float(np.sum(weights * x)) if np.sum(weights) > 0 else None

        mean_sensitivity = wavg(sens)
        mean_specificity = wavg(spec)
        mean_precision   = wavg(prec)
        mean_recall      = wavg(sens)
        mean_f1          = wavg(f1)

    return accuracy, mean_specificity, mean_sensitivity, mean_precision, mean_recall, mean_f1

def _nan_to_none(x):
    try:
        return None if (x is None or np.isnan(x)) else float(x)
    except TypeError:
        return None

def sum_confusion_matrices(conf_matrices):
    overall_conf_matrix = np.sum(conf_matrices, axis=0)
    return overall_conf_matrix

def positive_class_scores(model, X, n_classes=2):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1] if n_classes == 2 else proba
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if n_classes == 2:
            return scores if scores.ndim == 1 else scores[:, 1]
        return scores
    raise RuntimeError("Model does not expose predict_proba or decision_function.")


def get_param_grid(classify_method):
    searchers = {
        'dt': {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [None, 10, 20, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5]
        },
        'rf': {
            'n_estimators': [200, 300, 500],
            'max_depth': [None, 10, 20, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_features': ['sqrt', 'log2', None, 0.2, 0.5, 1.0],
            'bootstrap': [True, False]
        },
        'svm': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto']
        },
        'lr': {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga']
        },
        'knn': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        },
        'gb': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'max_depth': [2, 3, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5]
        }
    }
    return searchers[classify_method]

def _unique_groups_index(groups):
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    return uniq, inv

def _derive_group_truth(y_true, groups, strategy='strict'):
    y_true = np.asarray(y_true)
    uniq, inv = _unique_groups_index(groups)
    y_group = np.empty(len(uniq), dtype=y_true.dtype)
    for g_idx in range(len(uniq)):
        vals = y_true[inv == g_idx]
        if strategy == 'strict' and np.unique(vals).size == 1:
            y_group[g_idx] = vals[0]
        else:
            y_group[g_idx] = Counter(vals).most_common(1)[0][0]
    return uniq, y_group

def _aggregate_group_scores(y_score, groups, n_classes, agg='mean'):
    groups = np.asarray(groups)
    uniq, inv = _unique_groups_index(groups)

    y_score = np.asarray(y_score)
    if n_classes == 2 and y_score.ndim == 1:
        probs = y_score.reshape(-1, 1)
        C = 1
    else:
        probs = y_score
        C = probs.shape[1]

    agg_scores = np.zeros((len(uniq), C), dtype=float)

    for g_idx in range(len(uniq)):
        mask = (inv == g_idx)
        if agg == 'mean':
            agg_scores[g_idx] = probs[mask].mean(axis=0)
        elif agg == 'median':
            agg_scores[g_idx] = np.median(probs[mask], axis=0)
        elif agg == 'vote':
            if C == 1:
                votes = (probs[mask] >= 0.5).astype(int).ravel()
                agg_scores[g_idx] = votes.mean()
            else:
                preds = np.argmax(probs[mask], axis=1)
                freq = np.bincount(preds, minlength=C) / preds.size
                agg_scores[g_idx] = freq
        else:
            raise ValueError(f"Unknown agg '{agg}'. Use 'mean' | 'median' | 'vote'.")
    return uniq, agg_scores.squeeze()

def evaluate_group_level(y_true, y_score, groups, unique_labels_encoded,
                         agg='mean', threshold=0.8, proportion=0.3):

    n_classes = len(unique_labels_encoded)

    g_ids_truth, y_true_g = _derive_group_truth(y_true, groups, strategy='strict')

    if agg in ('mean', 'median', 'vote'):
        g_ids_pred, g_scores = _aggregate_group_scores(y_score, groups, n_classes, agg=agg)

    elif agg == 'max':
        uniq, inv = _unique_groups_index(groups)
        y_score = np.asarray(y_score)
        if n_classes == 2 and y_score.ndim == 1:
            g_scores = np.zeros(len(uniq))
            for g_idx in range(len(uniq)):
                g_scores[g_idx] = y_score[inv == g_idx].max()
        else:
            g_scores = np.zeros((len(uniq), n_classes))
            for g_idx in range(len(uniq)):
                g_scores[g_idx] = y_score[inv == g_idx].max(axis=0)
        g_ids_pred = uniq

    elif agg == 'proportion':
        if n_classes != 2:
            raise ValueError("agg='proportion' is only defined for binary classification.")
        uniq, inv = _unique_groups_index(groups)
        y_score = np.asarray(y_score)

        g_scores = np.zeros(len(uniq))
        for g_idx in range(len(uniq)):
            probs = y_score[inv == g_idx]
            if probs.ndim > 1:
                probs = probs[:, 1]
            frac_pos = np.mean(probs >= threshold)
            g_scores[g_idx] = frac_pos
        g_ids_pred = uniq

    else:
        raise ValueError(f"Unknown agg '{agg}'. Use 'mean' | 'median' | 'vote' | 'max' | 'proportion'.")

    assert np.array_equal(g_ids_truth, g_ids_pred), "Mismatch in group ids alignment."

    if n_classes == 2:
        if g_scores.ndim == 1:
            grp_cutoff = proportion if agg == 'proportion' else threshold
            y_pred_g = (g_scores >= grp_cutoff).astype(int)
            y_score_out = g_scores  # keep the scalar group score
        else:
            y_pred_g = np.argmax(g_scores, axis=1)
            y_score_out = g_scores[:, 1]
    else:
        y_pred_g = np.argmax(g_scores, axis=1)
        y_score_out = g_scores

    conf_g = confusion_matrix(y_true_g, y_pred_g, labels=unique_labels_encoded)
    acc, spec, sens, prec, rec, f1 = calculate_metrics(conf_g)

    return {
        "accuracy": acc,
        "specificity": spec,
        "sensitivity": sens,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "conf_matrix": conf_g,
        "y_true": y_true_g,
        "y_score": y_score_out,
        "group_ids": g_ids_truth
    }

def combine_volume(results_list):
    all_keys = set()
    for r in results_list:
        all_keys.update(vars(r.volume).keys())

    out = {}
    for k in all_keys:
        vals = []
        for r in results_list:
            v = getattr(r.volume, k, None)
            if v is None:
                continue

            if isinstance(v, (int, np.integer)):
                vals.append(int(v))
        if vals:
            out[k] = max(vals)

    return make_namespace(out)

def combine_results(results):
    first = results[0]

    metrics = ["accuracy", "precision", "recall",
               "specificity", "sensitivity", "f1"]

    agg_metrics = {}
    for m in metrics:
        vals = [
            getattr(r.results, m, None)
            for r in results
            if getattr(r.results, m, None) is not None
        ]
        arr = np.array(vals, dtype=float)
        if arr.size == 0:
            return None
        agg_metrics[m] = float(np.nanmean(arr))

    conf_mats = [
        r.meta.conf_matrix
        for r in results
        if getattr(r.meta, "conf_matrix", None) is not None
    ]
    if conf_mats:
        combined_conf_mat = np.sum(conf_mats, axis=0)
    else:
        combined_conf_mat = None

    y_trues = [
        np.asarray(r.meta.y_true)
        for r in results
        if getattr(r.meta, "y_true", None) is not None
    ]
    y_scores = [
        np.asarray(r.meta.y_score)
        for r in results
        if getattr(r.meta, "y_score", None) is not None
    ]
    test_indices = [
        np.asarray(r.meta.test_index)
        for r in results
        if getattr(r.meta, "test_index", None) is not None
    ]

    merged_test_meta = {}
    if getattr(first.meta, "test_meta", None) is not None:
        for key in first.meta.test_meta.keys():
            merged_test_meta[key] = np.concatenate(
                [
                    np.asarray(r.meta.test_meta[key])
                    for r in results
                    if getattr(r.meta, "test_meta", None) is not None
                       and key in r.meta.test_meta
                ],
                axis=0
            )

    feat_imps = [
        np.asarray(r.meta.feature_importances)
        for r in results
        if getattr(r.meta, "feature_importances", None) is not None
    ]
    if feat_imps:
        shapes = {fi.shape for fi in feat_imps}
        if len(shapes) == 1:
            combined_feat_imp = np.mean(feat_imps, axis=0)
        else:
            combined_feat_imp = None
    else:
        combined_feat_imp = None

    combined_meta_ns = make_namespace({
        "conf_matrix": combined_conf_mat,
        "y_true": np.concatenate(y_trues, axis=0) if y_trues else None,
        "y_score": np.concatenate(y_scores, axis=0) if y_scores else None,
        "feature_importances": combined_feat_imp,
        "test_index": np.concatenate(test_indices, axis=0) if test_indices else None,
        "test_meta": merged_test_meta if merged_test_meta else None,
        "model": results[0].params.model,
    })

    combined_params_ns = make_namespace({
        **vars(first.params),
        "mode": f"{getattr(first.params, 'mode', 'sample')}_combined",
        "num_results": len(results),
    })

    combined_volume_ns = make_namespace({
        **vars(first.volume)
    })

    combined_results_ns = make_namespace({
        "accuracy":     agg_metrics["accuracy"],
        "precision":    agg_metrics["precision"],
        "recall":       agg_metrics["recall"],
        "specificity":  agg_metrics["specificity"],
        "sensitivity":  agg_metrics["sensitivity"],
        "f1":           agg_metrics["f1"],
    })

    combined_res = Result(
        set_params = combined_params_ns,
        set_volume = combined_volume_ns,
        set_results = combined_results_ns,
        set_meta = combined_meta_ns
    )

    return combined_res

def aggregate_result(
    res,
    agg_group,
    agg_method = "mean",
    threshold = 0.5,
    proportion = None
):

    y_true  = np.asarray(res.results.y_true)
    y_score = np.asarray(res.results.y_score)

    test_meta = getattr(res.results, "test_meta", None)
    groups_test = np.asarray(test_meta[agg_group])

    unique_labels_encoded = np.unique(y_true)

    group_eval = evaluate_group_level(
        y_true=y_true,
        y_score=y_score,
        groups=groups_test,
        unique_labels_encoded=unique_labels_encoded,
        agg=agg_method,
        threshold=threshold,
        proportion=proportion
    )

    new_params = make_namespace({
        **vars(res.params),
        "mode": "group",
        "agg_group": agg_group,
        "group_agg": agg_method,
        "group_threshold": threshold,
        "group_proportion": proportion
    })

    return Result(
        set_params=new_params,
        set_volume=res.volume,
        set_results=make_namespace(group_eval),
    )