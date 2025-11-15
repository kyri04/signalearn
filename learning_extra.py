from signalearn.preprocess import sample
from signalearn.learning import *
from signalearn.learning_utility import combine_results, calculate_metrics
from sklearn.metrics import confusion_matrix
import numpy as np

def ordinal_classify(
    points,
    y_attr,
    target,
    group=None,
    model=RandomForestClassifier(),
    test_size=0.2,
    split_state=42,
    scaler=None,
    sampler=None
):
    classes = sorted(find_unique(points, target))
    thresholds = classes[1:]
    results = {}

    for thr in thresholds:
        pos_vals = [str(c) for c in classes if c >= thr]
        pos_points = set(filter(points, target, pos_vals))

        ord_attr = f"ordinal_{target}_{thr}"

        pos_label = f"{target}>={thr}"
        neg_label = f"{target}<{thr}"

        for p in points:
            setattr(p, ord_attr, pos_label if p in pos_points else neg_label)

        res = classify(
            points,
            y_attr=y_attr,
            target=ord_attr,
            group=group,
            model=model,
            test_size=test_size,
            split_state=split_state,
            scaler=scaler,
            sampler=sampler
        )

        results[thr] = res

    return results

def shuffle_learn(
    points, 
    target, 
    y_attr,
    learn_func,
    model,
    group=None,  
    test_size=0.2,
    shuffles=5,
    scaler=None,
    sampler=None
):
    results = []
    for rs in range(shuffles):
        results.append(learn_func(
            points=points, 
            y_attr=y_attr,
            target = target, 
            group = group,
            model = model, 
            test_size = test_size,  
            split_state = rs,
            scaler=scaler,
            sampler=sampler
        ))
        
    return results

def attr_curve(
    points, 
    y_attr,
    target, 
    by_attribute,
    learn_func,
    model,
    group=None, 
    test_size=0.2,
    split_state=42,
    divisions=5,
    start_val=0,
    shuffles_per_split=None,
    scaler=None,
    sampler=None
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
                model=model,
                test_size=test_size,
                shuffles=shuffles_per_split,
                scaler=scaler,
                sampler=sampler
            )
            results.append(combine_results(res_list))
            
        else:
            res = learn_func(
                points=subset,
                y_attr=y_attr,
                target=target,
                group=group,
                model=model,
                test_size=test_size,
                split_state=split_state,
                scaler=scaler,
                sampler=sampler
            )
            results.append(res)

    return results

def data_curve(
    points, 
    target, 
    y_attr,
    learn_func,
    model,
    group=None,  
    test_size=0.2,
    split_state=42, 
    divisions=5,
    start_fraction=0.05,
    shuffles_per_split=None,
    scaler=None,
    sampler=None
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
                model=model,
                test_size=test_size,
                shuffles=shuffles_per_split,
                scaler=scaler,
                sampler=sampler
            )
            res = combine_results(res_list)
            results.append(res)
            
        else:
            res = learn_func(
                points=subset,
                y_attr=y_attr,
                target=target,
                group=group,
                model=model,
                test_size=test_size,
                split_state=split_state,
                scaler=scaler,
                sampler=sampler
            )
            results.append(res)

    return results
