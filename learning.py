from signalearn.utility import *
from signalearn.classes import *
from signalearn.learning_utility import build_feature_matrix, prepare_labels, prepare_groups, get_single_split
import numpy as np

def split(
    dataset,
    group=None,
    test_size=0.2,
    seed=42
):
    N = len(dataset.samples)
    groups = prepare_groups(group)
    train_idx, test_idx = get_single_split(N, None, groups, test_size, seed)
    if groups is not None:
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))

    train = Dataset([dataset.samples[i] for i in train_idx])
    test = Dataset([dataset.samples[i] for i in test_idx])
    return Split(train=train, test=test)

def classify(
    x,
    target,
    model
):
    X, _, _ = build_feature_matrix(as_fields(x))
    y = prepare_labels(target)
    model.fit(X, y)
    return model

def regress(
    x,
    target,
    model
):
    X, _, _ = build_feature_matrix(as_fields(x))
    y = np.array([f.values for f in target.fields], dtype=float)
    model.fit(X, y)
    return model

def cluster(
    y,
    model,
):
    y_fields = as_fields(y)
    dataset = y_fields[0]._dataset
    X, _, _ = build_feature_matrix(y_fields)
    labels = model.fit_predict(X)
    name = snake(model.__class__.__name__)
    samples = []
    for sample, label in zip(dataset.samples, labels):
        new = new_sample(sample, {name: label})
        new.fields[name].label = pretty(name)
        samples.append(new)
    return Dataset(samples)

def reduce(
    y,
    model,
):
    y_fields = as_fields(y)
    dataset = y_fields[0]._dataset
    X, _, _ = build_feature_matrix(y_fields)
    Z = model.fit_transform(X)
    prefix = snake(model.__class__.__name__)
    samples = []
    for i, sample in enumerate(dataset.samples):
        updates = {}
        for j in range(Z.shape[1]):
            attr = f"{prefix}{j+1}"
            updates[attr] = Z[i, j]
        new = new_sample(sample, updates)
        for j in range(Z.shape[1]):
            attr = f"{prefix}{j+1}"
            new.fields[attr].label = pretty(attr)
        samples.append(new)
    return Dataset(samples)
