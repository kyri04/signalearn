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

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

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
        "lr": LogisticRegression(random_state=42, class_weight='balanced', max_iter=10000),
        "knn": KNeighborsClassifier(),
        "gb": GradientBoostingClassifier(random_state=42),
    }
    return classifiers[method]

def get_regressor(method):
    regressors = {
        "dt": DecisionTreeRegressor(random_state=42),
        "rf": RandomForestRegressor(random_state=42, n_estimators=300, max_depth=10),
        "svm": SVR(),
        "lr": LinearRegression(),
        "knn": KNeighborsRegressor(),
        "gb": GradientBoostingRegressor(random_state=42),
    }
    return regressors[method]

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

def unique_groups_index(groups):
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    return uniq, inv

def derive_group_truth(y_true, groups, strategy='strict'):
    y_true = np.asarray(y_true)
    uniq, inv = unique_groups_index(groups)
    y_group = np.empty(len(uniq), dtype=y_true.dtype)
    for g_idx in range(len(uniq)):
        vals = y_true[inv == g_idx]
        if strategy == 'strict' and np.unique(vals).size == 1:
            y_group[g_idx] = vals[0]
        else:
            y_group[g_idx] = Counter(vals).most_common(1)[0][0]
    return uniq, y_group

def aggregate_group_scores(y_score, groups, n_classes, agg='mean'):
    groups = np.asarray(groups)
    uniq, inv = unique_groups_index(groups)

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

def evaluate_group_level_classification(y_true, y_score, groups, unique_labels_encoded,
                         agg='mean', threshold=0.8, proportion=0.3):

    n_classes = len(unique_labels_encoded)

    g_ids_truth, y_true_g = derive_group_truth(y_true, groups, strategy='strict')

    if agg in ('mean', 'median', 'vote'):
        g_ids_pred, g_scores = aggregate_group_scores(y_score, groups, n_classes, agg=agg)

    elif agg == 'max':
        uniq, inv = unique_groups_index(groups)
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
        uniq, inv = unique_groups_index(groups)
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

def aggregate_group_values(y, groups, agg='mean', param=None):
    y = np.asarray(y, dtype=float).ravel()
    uniq, inv = unique_groups_index(groups)
    out = np.empty(len(uniq), dtype=float)
    for g_idx in range(len(uniq)):
        vals = y[inv == g_idx]
        if vals.size == 0:
            out[g_idx] = np.nan
        elif agg == 'mean':
            out[g_idx] = float(np.mean(vals))
        elif agg == 'median':
            out[g_idx] = float(np.median(vals))
        elif agg == 'max':
            out[g_idx] = float(np.max(vals))
        elif agg == 'percentile':
            q = param if param is not None else 90
            out[g_idx] = np.percentile(vals, q)
        else:
            raise ValueError("Unknown agg for regression. Use 'mean' | 'median' | 'max'.")
    return uniq, out

def evaluate_group_level_regression(y_true, y_pred, groups, agg='mean', param=None):
    uniq_t, y_true_g = aggregate_group_values(y_true, groups, agg=agg, param=param)
    uniq_p, y_pred_g = aggregate_group_values(y_pred, groups, agg=agg, param=param)
    assert np.array_equal(uniq_t, uniq_p), "Mismatch in group ids alignment."

    diff = y_pred_g - y_true_g
    m = np.isfinite(diff) & np.isfinite(y_true_g)
    yt, yp, df = y_true_g[m], y_pred_g[m], diff[m]
    if yt.size == 0:
        mae = mse = rmse = r2 = np.nan
    else:
        mae  = float(np.mean(np.abs(df)))
        mse  = float(np.mean(df**2))
        rmse = float(np.sqrt(mse))
        ss_res = float(np.sum(df**2))
        ss_tot = float(np.sum((yt - yt.mean())**2))
        r2 = float(1.0 - ss_res/ss_tot) if ss_tot > 0 else float("nan")

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "y_true": y_true_g,
        "y_pred": y_pred_g,
        "group_ids": uniq_t
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

    cls_metrics = {"accuracy","precision","recall","specificity","sensitivity","f1"}
    reg_metrics = {"mae","mse","rmse","r2"}
    all_metrics = cls_metrics | reg_metrics

    def mean_or_none(vals):
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.nanmean(vals)) if vals else None

    agg = {}
    for m in all_metrics:
        agg[m] = mean_or_none([getattr(r.results, m, None) for r in results])
    agg = {k:v for k,v in agg.items() if v is not None}

    conf_mats = [getattr(r.meta, "conf_matrix", None) for r in results]
    conf_mats = [cm for cm in conf_mats if cm is not None]
    combined_conf_mat = np.sum(conf_mats, axis=0) if conf_mats else None

    y_trues   = [np.asarray(getattr(r.meta,"y_true"))   for r in results if getattr(r.meta,"y_true",None)   is not None]
    y_scores  = [np.asarray(getattr(r.meta,"y_score"))  for r in results if getattr(r.meta,"y_score",None)  is not None]
    y_preds   = [np.asarray(getattr(r.meta,"y_pred"))   for r in results if getattr(r.meta,"y_pred",None)   is not None]
    residuals = [np.asarray(getattr(r.meta,"residuals"))for r in results if getattr(r.meta,"residuals",None)is not None]
    test_idx  = [np.asarray(getattr(r.meta,"test_index")) for r in results if getattr(r.meta,"test_index",None) is not None]

    if y_trues and y_preds:
        yt = np.concatenate(y_trues, axis=0)
        yp = np.concatenate(y_preds,  axis=0)
        mae  = float(np.mean(np.abs(yt - yp)))
        mse  = float(np.mean((yt - yp)**2))
        rmse = float(np.sqrt(mse))
        ss_tot = float(np.sum((yt - yt.mean())**2))
        r2   = float(1.0 - np.sum((yt - yp)**2)/ss_tot) if ss_tot > 0 else float("nan")
        agg.update({"mae": mae, "mse": mse, "rmse": rmse, "r2": r2})

    merged_test_meta = {}
    keys = set()
    for r in results:
        tm = getattr(r.meta, "test_meta", None)
        if isinstance(tm, dict): keys.update(tm.keys())
    for k in keys:
        merged_test_meta[k] = np.concatenate(
            [np.asarray(r.meta.test_meta[k]) for r in results
             if getattr(r.meta, "test_meta", None) is not None and k in r.meta.test_meta],
            axis=0
        ) if any(getattr(r.meta, "test_meta", None) is not None and k in r.meta.test_meta for r in results) else None

    feat_imps = [np.asarray(getattr(r.meta,"feature_importances")) for r in results
                 if getattr(r.meta,"feature_importances",None) is not None]
    if feat_imps and len({fi.shape for fi in feat_imps}) == 1:
        combined_feat_imp = np.mean(feat_imps, axis=0)
    else:
        combined_feat_imp = None

    vol0 = vars(first.volume)
    combined_volume = {}
    for k in vol0.keys():
        vals = [getattr(r.volume, k, None) for r in results]
        if all(isinstance(v, (int,float,np.number)) for v in vals if v is not None):
            combined_volume[k] = float(np.nansum([float(v) for v in vals if v is not None]))
        else:
            combined_volume[k] = vol0[k]

    combined_params = {
        **vars(first.params),
        "mode": f"{getattr(first.params, 'mode', 'combined')}_combined",
        "num_results": len(results),
    }

    combined_meta = {
        "conf_matrix": combined_conf_mat,
        "y_true":   np.concatenate(y_trues,  axis=0) if y_trues  else None,
        "y_score":  np.concatenate(y_scores, axis=0) if y_scores else None,
        "y_pred":   np.concatenate(y_preds,  axis=0) if y_preds  else None,
        "residuals":np.concatenate(residuals,axis=0) if residuals else None,
        "feature_importances": combined_feat_imp,
        "test_index": np.concatenate(test_idx, axis=0) if test_idx else None,
        "test_meta": merged_test_meta or None,
        "model": None,
    }

    combined = Result(
        set_params = combined_params,
        set_volume = combined_volume,
        set_results = agg,
        set_meta = combined_meta
    )
    return combined

def aggregate_result(
    res,
    agg_group,
    agg_method="mean",
    threshold=0.5,
    proportion=None,
    param=None
):
    # pull arrays from META (with legacy fallbacks)
    y_true  = np.asarray(getattr(res.meta, "y_true",
                        getattr(res.results, "y_true", [])))
    y_score = getattr(res.meta, "y_score",
              getattr(res.results, "y_score", None))
    y_pred  = getattr(res.meta, "y_pred", None)

    test_meta = getattr(res.meta, "test_meta",
                getattr(res.results, "test_meta", None))
    if test_meta is None or agg_group not in test_meta:
        raise KeyError(f"agg_group '{agg_group}' not found in test_meta")
    groups_test = np.asarray(test_meta[agg_group])

    # classification if we have class info or scores; else regression
    unique_labels_encoded = getattr(res.params, "unique_labels", None)
    is_classification = (unique_labels_encoded is not None) or (y_score is not None)

    if is_classification:
        # Always derive labels from the y_true actually present here
        y_true = np.asarray(y_true)
        unique_labels_encoded = np.unique(y_true)

        if y_score is None and y_pred is not None:
            y_score = np.asarray(y_pred, dtype=float)

        group_eval = evaluate_group_level_classification(
            y_true=y_true,
            y_score=y_score,
            groups=groups_test,
            unique_labels_encoded=unique_labels_encoded,
            agg=agg_method,
            threshold=threshold,
            proportion=proportion
        )
        # split metrics vs meta for Result
        results_dict = {
            k: group_eval[k] for k in
            ("accuracy","specificity","sensitivity","precision","recall","f1")
            if k in group_eval
        }
        meta_dict = {
            "conf_matrix": group_eval.get("conf_matrix"),
            "y_true": group_eval.get("y_true"),
            "y_score": group_eval.get("y_score"),
            "group_ids": group_eval.get("group_ids"),
        }

    else:
        if y_pred is None:
            raise ValueError("Regression aggregation requires meta.y_pred.")
        group_eval = evaluate_group_level_regression(
            y_true=y_true,
            y_pred=np.asarray(y_pred, dtype=float),
            groups=groups_test,
            agg=agg_method,  # 'mean' | 'median' | 'max'
            param=param
        )
        results_dict = {
            k: group_eval[k] for k in ("mae","mse","rmse","r2") if k in group_eval
        }
        meta_dict = {
            "y_true": group_eval.get("y_true"),
            "y_pred": group_eval.get("y_pred"),
            "group_ids": group_eval.get("group_ids"),
        }

    new_params = {
        **vars(res.params),
        "mode": "group",
        "agg_group": agg_group,
        "group_agg": agg_method,
        "group_threshold": threshold,
        "group_proportion": proportion,
    }

    return Result(
        set_params=new_params,          # dicts, not SimpleNamespace
        set_volume=vars(res.volume),    # keep original volume
        set_results=results_dict,
        set_meta=meta_dict
    )

import numpy as np
from signalearn import learning_utility


def optimise_group_sens_spec(
    res,
    agg_group="id_sample",
    agg_method="proportion",
    n_steps=21,
):
    """
    Grid-search threshold and proportion in [0, 1] to maximise
    a combined sensitivity/specificity score.

    Score used: (sensitivity + specificity) / 2  (balanced accuracy)

    Parameters
    ----------
    res : Result
        Base per-scan Result object (e.g. results[0]).
    agg_group : str
        Grouping key, e.g. 'id_sample'.
    agg_method : str
        Aggregation method, e.g. 'proportion'.
    n_steps : int
        Number of grid steps per axis (default 21 -> ~0.05 step).

    Returns
    -------
    best_result : Result
        Result from learning_utility.aggregate_result with best score.
    best_params : dict
        Dict with 'threshold', 'proportion', 'sensitivity',
        'specificity', and 'score'.
    """
    proportions = np.linspace(0.0, 1.0, n_steps)
    thresholds  = np.linspace(0.0, 1.0, n_steps)

    best_score = None
    best_result = None
    best_p = None
    best_t = None
    best_sens = None
    best_spec = None

    for p in proportions:
        if p <= 0.0 or p >= 1.0:
            continue
        for t in thresholds:
            if t <= 0.0 or t >= 1.0:
                continue

            try:
                agg_res = learning_utility.aggregate_result(
                    res,
                    agg_group=agg_group,
                    agg_method=agg_method,
                    threshold=t,
                    proportion=p,
                )
            except Exception:
                # skip combinations that fail aggregation
                continue

            sens = getattr(agg_res.results, "sensitivity", None)
            spec = getattr(agg_res.results, "specificity", None)

            if sens is None or spec is None:
                continue

            score = 0.5 * (sens + spec)  # balanced accuracy

            if (best_score is None) or (score > best_score):
                best_score = score
                best_result = agg_res
                best_p = p
                best_t = t
                best_sens = sens
                best_spec = spec

    if best_result is None:
        raise ValueError("No valid (threshold, proportion) pair produced sensitivity/specificity.")

    best_params = {
        "threshold": best_t,
        "proportion": best_p,
        "sensitivity": best_sens,
        "specificity": best_spec,
        "score": best_score,
    }
    return best_result, best_params
