from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

from collections import Counter
from signalearn.classes import ClassificationResult

import numpy as np
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
    else:
        raise ValueError("label must be either a string or a list")
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
        "rf": RandomForestClassifier(random_state=42, n_estimators=300, max_depth=10),
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

    # Per-class components
    tps = np.diag(conf_matrix)
    fns = conf_matrix.sum(axis=1) - tps  # row sums minus tp
    fps = conf_matrix.sum(axis=0) - tps  # col sums minus tp
    tns = total_samples - (tps + fns + fps)

    # Avoid zero-division: compute with masking
    with np.errstate(divide='ignore', invalid='ignore'):
        sens = np.where((tps + fns) > 0, tps / (tps + fns), np.nan)   # recall / TPR
        spec = np.where((tns + fps) > 0, tns / (tns + fps), np.nan)   # TNR
        prec = np.where((tps + fps) > 0, tps / (tps + fps), np.nan)   # PPV
        f1 = np.where(
            (prec + sens) > 0,
            2 * (prec * sens) / (prec + sens),
            np.nan
        )

    if average == "binary":
        # Return metrics for the designated positive class only
        i = pos_index
        mean_sensitivity = _nan_to_none(sens[i])
        mean_specificity = _nan_to_none(spec[i])  # specificity wrt positive class = TN/(TN+FP) for that class
        mean_precision  = _nan_to_none(prec[i])
        mean_recall     = _nan_to_none(sens[i])
        mean_f1         = _nan_to_none(f1[i])

    else:
        # supports = actual class counts (row sums)
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

def fit_model(X_train, y_train, classifier, tune):
    if tune:
        base = get_classifier(classifier)
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', base)])
        grid = get_param_grid(classifier)
        grid_piped = {f'clf__{k}': v for k, v in grid.items()}
        search = RandomizedSearchCV(pipe, param_distributions=grid_piped, n_iter=20, scoring='accuracy', cv=3, random_state=42, n_jobs=-1, verbose=1)
        search.fit(X_train, y_train)
        print(search.best_params_)
        return search.best_estimator_
    model = get_classifier(classifier)
    model.fit(X_train, y_train)
    return model

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

def combine_results(results_list):
    """
    Combine a list of ClassificationResult objects (e.g., from shuffle_classify) into one.
    - Sums confusion matrices across runs and recomputes metrics.
    - Concatenates y_true / y_score across runs (handles binary & multiclass).
    - Summarizes params per key:
        * constant -> single value
        * numeric  -> (min, max)
        * other    -> "{a, b, ...}"
    - Does the same aggregation for group_results when present.
    """
    if not results_list:
        raise ValueError("combine_results expects a non-empty list of results.")

    # ---- Summarize params across runs ----
    all_param_keys = set()
    for r in results_list:
        all_param_keys.update(vars(r.params).keys())

    param_values = {k: [] for k in all_param_keys}
    for r in results_list:
        rp = vars(r.params)
        for k in all_param_keys:
            param_values[k].append(rp.get(k, None))

    def _is_number(x):
        return isinstance(x, (int, float, np.integer, np.floating))

    def _all_equal(xs):
        if not xs:
            return True
        first = xs[0]
        for v in xs[1:]:
            if isinstance(v, np.ndarray) and isinstance(first, np.ndarray):
                if not np.array_equal(v, first):
                    return False
            else:
                if v != first:
                    return False
        return True

    def _summarize(vals):
        xs = [v for v in vals if v is not None]
        if len(xs) == 0:
            return None
        if _all_equal(xs):
            return xs[0]
        if all(_is_number(v) for v in xs):
            return (min(xs), max(xs))
        uniq = sorted({str(v) for v in xs})
        return "{" + ", ".join(uniq) + "}"

    summarized_params = {k: _summarize(vs) for k, vs in param_values.items()}

    # Ensure unique_labels present; if different across runs, make a union preserving first order
    if "unique_labels" not in summarized_params or summarized_params["unique_labels"] is None:
        if hasattr(results_list[0].params, "unique_labels"):
            summarized_params["unique_labels"] = results_list[0].params.unique_labels

    try:
        first_ul = np.array(results_list[0].params.unique_labels, dtype=object)
        unions = list(first_ul)
        seen = set(unions)
        for r in results_list[1:]:
            ul = getattr(r.params, "unique_labels", None)
            if ul is None:
                continue
            for x in ul:
                if x not in seen:
                    unions.append(x)
                    seen.add(x)
        summarized_params["unique_labels"] = np.array(unions, dtype=object)
    except Exception:
        pass

    params_ns = make_namespace(summarized_params)

    # ---- Aggregate sample-level results ----
    confs = [r.results.conf_matrix for r in results_list if getattr(r.results, "conf_matrix", None) is not None]
    if not confs:
        raise ValueError("No confusion matrices found in results_list.")
    overall_conf = sum_confusion_matrices(confs)
    acc, spec, sens, prec, rec, f1 = calculate_metrics(overall_conf)

    # Concatenate y_true
    y_true_parts = [r.results.y_true for r in results_list if getattr(r.results, "y_true", None) is not None]
    y_true_all = np.concatenate(y_true_parts) if y_true_parts else None

    # Concatenate y_score (binary 1D vs multiclass 2D)
    y_score_parts = [r.results.y_score for r in results_list if getattr(r.results, "y_score", None) is not None]
    y_score_all = None
    if y_score_parts:
        n_classes = overall_conf.shape[0]
        if n_classes == 2:
            normalized = []
            for s in y_score_parts:
                s = np.asarray(s)
                if s.ndim == 1:
                    normalized.append(s)
                elif s.ndim == 2 and s.shape[1] == 2:
                    normalized.append(s[:, 1])
                else:
                    normalized.append(s.reshape(s.shape[0], -1)[:, -1])
            y_score_all = np.concatenate(normalized)
        else:
            normalized = []
            for s in y_score_parts:
                s = np.asarray(s)
                if s.ndim == 1:
                    s = s.reshape(-1, 1)
                if s.shape[1] != n_classes:
                    tmp = np.zeros((s.shape[0], n_classes), dtype=float)
                    m = min(n_classes, s.shape[1])
                    tmp[:, :m] = s[:, :m]
                    s = tmp
                normalized.append(s)
            y_score_all = np.vstack(normalized)

    results_ns = make_namespace({
        "accuracy":    acc,
        "specificity": spec,
        "sensitivity": sens,
        "precision":   prec,
        "recall":      rec,
        "f1":          f1,
        "conf_matrix": overall_conf,
        "y_true":      y_true_all,
        "y_score":     y_score_all,
    })

    # ---- Aggregate group-level results (if present) ----
    group_results_available = any(getattr(r, "group_results", None) is not None for r in results_list)
    group_ns = None
    if group_results_available:
        g_confs = [r.group_results.conf_matrix for r in results_list if getattr(r.group_results, "conf_matrix", None) is not None]
        if g_confs:
            overall_gconf = sum_confusion_matrices(g_confs)
            g_acc, g_spec, g_sens, g_prec, g_rec, g_f1 = calculate_metrics(overall_gconf)
        else:
            overall_gconf = None
            g_acc = g_spec = g_sens = g_prec = g_rec = g_f1 = None

        # Concatenate group y_true / y_score / group_ids when available
        g_ytrue_parts = [r.group_results.y_true for r in results_list if getattr(r.group_results, "y_true", None) is not None]
        g_ytrue_all = np.concatenate(g_ytrue_parts) if g_ytrue_parts else None

        g_score_parts = [r.group_results.y_score for r in results_list if getattr(r.group_results, "y_score", None) is not None]
        g_score_all = None
        if g_score_parts:
            n_classes = overall_conf.shape[0] if overall_conf is not None else len(params_ns.unique_labels)
            if n_classes == 2:
                normalized = []
                for s in g_score_parts:
                    s = np.asarray(s)
                    if s.ndim == 1:
                        normalized.append(s)
                    elif s.ndim == 2 and s.shape[1] == 2:
                        normalized.append(s[:, 1])
                    else:
                        normalized.append(s.reshape(s.shape[0], -1)[:, -1])
                g_score_all = np.concatenate(normalized)
            else:
                normalized = []
                for s in g_score_parts:
                    s = np.asarray(s)
                    if s.ndim == 1:
                        s = s.reshape(-1, 1)
                    if s.shape[1] != n_classes:
                        tmp = np.zeros((s.shape[0], n_classes), dtype=float)
                        m = min(n_classes, s.shape[1])
                        tmp[:, :m] = s[:, :m]
                        s = tmp
                    normalized.append(s)
                g_score_all = np.vstack(normalized)

        g_ids_parts = [r.group_results.group_ids for r in results_list if getattr(r.group_results, "group_ids", None) is not None]
        g_ids_all = np.concatenate(g_ids_parts) if g_ids_parts else None

        group_ns = make_namespace({
            "accuracy":    g_acc,
            "specificity": g_spec,
            "sensitivity": g_sens,
            "precision":   g_prec,
            "recall":      g_rec,
            "f1":          g_f1,
            "conf_matrix": overall_gconf,
            "y_true":      g_ytrue_all,
            "y_score":     g_score_all,
            "group_ids":   g_ids_all,
        })

    # ---- Return combined ClassificationResult ----
    return ClassificationResult(
        set_params=params_ns,
        set_results=results_ns,
        set_group_results=group_ns
    )