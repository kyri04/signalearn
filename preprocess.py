import random
import numpy as np
from collections import defaultdict
from scipy.interpolate import interp1d
from signalearn.classes import Dataset, Sample
from signalearn.utility import new_sample

def sample(dataset, f=0.05):
    sample_size = int(len(dataset) * f)
    sampled = random.sample(dataset.samples, sample_size)
    return Dataset([new_sample(s) for s in sampled])

def interpolate(x, y, n=50):
    dataset = x._dataset
    x_name = x.name
    y_name = y.name
    samples = []
    for sample, x_field, y_field in zip(dataset.samples, x.fields, y.fields):
        x_vals = np.asarray(x_field.values, dtype=float)
        y_vals = np.asarray(y_field.values, dtype=float)
        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        x_vals, y_vals = x_vals[mask], y_vals[mask]
        if x_vals.size < 2:
            continue
        f = interp1d(x_vals, y_vals, kind='linear', fill_value="extrapolate")
        x_uniform = np.linspace(x_vals[0], x_vals[-1], n)
        y_uniform = f(x_uniform)
        samples.append(new_sample(sample, {x_name: x_uniform, y_name: y_uniform}))
    return Dataset(samples)

def trim(x, amount, mode='both'):
    mode = mode.lower()
    dataset = x._dataset
    samples = []
    for sample, x_field in zip(dataset.samples, x.fields):
        x_vals = np.asarray(x_field.values, dtype=float)
        orig_len = x_vals.shape[0]
        left = x_vals[0]
        right = x_vals[-1]
        if mode in {'front', 'both'}:
            left += amount
        if mode in {'back', 'both'}:
            right -= amount
        mask = (x_vals >= left) & (x_vals <= right)
        updates = {}
        for name, field in sample.fields.items():
            arr = np.asarray(field.values)
            if arr.ndim >= 1 and arr.shape[0] == orig_len:
                updates[name] = arr[mask]
        samples.append(new_sample(sample, updates))
    return Dataset(samples)

def select(x, start, end):
    dataset = x._dataset
    samples = []
    for sample, x_field in zip(dataset.samples, x.fields):
        x_vals = np.asarray(x_field.values, dtype=float)
        orig_len = x_vals.shape[0]
        mask = (x_vals >= start) & (x_vals <= end)
        updates = {}
        for name, field in sample.fields.items():
            arr = np.asarray(field.values)
            if arr.ndim >= 1 and arr.shape[0] == orig_len:
                updates[name] = arr[mask]
        samples.append(new_sample(sample, updates))
    return Dataset(samples)

def resample(x, rate):
    rate = float(rate)
    step = 1.0 / rate
    eps = np.finfo(float).eps
    dataset = x._dataset
    x_name = x.name
    samples = []
    for sample, x_field in zip(dataset.samples, x.fields):
        x_arr = np.asarray(x_field.values, dtype=float).ravel()
        orig_len = x_arr.shape[0]
        mask = np.isfinite(x_arr)
        idx = np.flatnonzero(mask)
        x_valid = x_arr[idx]
        order = np.argsort(x_valid)
        idx = idx[order]
        x_valid = x_valid[order]
        if x_valid.size > 1:
            dup = np.concatenate(([True], np.diff(x_valid) > eps))
            if not np.all(dup):
                idx = idx[dup]
                x_valid = x_valid[dup]
        span = x_valid[-1] - x_valid[0]
        n_samples = max(2, int(np.floor(span * rate)) + 1)
        new_x = x_valid[0] + np.arange(n_samples) * step
        if new_x[-1] < x_valid[-1] - eps:
            new_x = np.append(new_x, x_valid[-1])
        updates = {x_name: new_x}
        for attr, field in sample.fields.items():
            if attr == x_name:
                continue
            arr = np.asarray(field.values)
            if arr.ndim == 0 or arr.shape[0] != orig_len:
                continue
            if not np.issubdtype(arr.dtype, np.number):
                continue
            arr_valid = arr[idx]
            interp_func = interp1d(
                x_valid,
                arr_valid,
                axis=0,
                kind='linear',
                bounds_error=False,
                fill_value="extrapolate"
            )
            updates[attr] = interp_func(new_x)
        samples.append(new_sample(sample, updates))
    return Dataset(samples)

def window(x, duration, overlap=0.0, allow_partial=False):
    dataset = x._dataset
    windows = []
    for sample, x_field in zip(dataset.samples, x.fields):
        x_vals = np.asarray(x_field.values, dtype=float).ravel()
        n = x_vals.size
        dx = np.diff(x_vals)
        dx = dx[np.isfinite(dx) & (dx > 0)]
        dt = float(np.median(dx))
        window_samples = max(1, int(round(duration / dt)))
        if window_samples <= 1:
            window_samples = 2
        step = max(1, int(round(window_samples * (1 - overlap))))
        if step <= 0:
            step = 1
        start = 0
        window_idx = 0
        while start < n:
            end = start + window_samples
            if end > n:
                if not allow_partial:
                    break
                end = n
            if end - start <= 1:
                break
            updates = {}
            for attr, field in sample.fields.items():
                arr = np.asarray(field.values)
                if arr.ndim >= 1 and arr.shape[0] == n:
                    updates[attr] = arr[start:end]
            parent = sample.fields.get("name")
            parent_id = parent.values if parent is not None else None
            updates["window_index"] = window_idx
            updates["name"] = f"{parent_id}_w{window_idx}"
            windows.append(new_sample(sample, updates))
            window_idx += 1
            start += step
    return Dataset(windows)

def func(y, func=np.log):
    dataset = y._dataset
    y_name = y.name
    samples = []
    for sample, y_field in zip(dataset.samples, y.fields):
        y_vals = y_field.values
        samples.append(new_sample(sample, {y_name: func(y_vals)}))
    return Dataset(samples)

def filter(x, val, includes=True):
    dataset = x._dataset
    if isinstance(val, str):
        vals = [val.lower()]
    else:
        vals = [v.lower() for v in val]

    samples = []
    for sample, field in zip(dataset.samples, x.fields):
        attr_val = str(field.values).lower()
        matched = any(v in attr_val for v in vals)
        if matched == includes:
            samples.append(new_sample(sample))
    return Dataset(samples)

def combine(match):
    dataset = match._dataset
    match_name = match.name
    groups = defaultdict(list)
    for sample, field in zip(dataset.samples, match.fields):
        k = field.values
        if k is not None:
            groups[str(k)].append(sample)

    out = []
    for k, grp in groups.items():
        if len(grp) < 2:
            continue

        params = {match_name: k}
        name_vals = []
        for p in grp:
            if "name" in p.fields:
                name_vals.append(str(p.fields["name"].values))
        params["name"] = "+".join(sorted(set(name_vals))) if name_vals else str(k)

        meta = {"name", match_name}
        attrs = set()
        for p in grp:
            attrs.update(a for a in p.fields.keys() if a not in meta)

        units_acc, labels_acc = {}, {}

        for a in sorted(attrs):
            vals = [p.fields[a].values for p in grp if a in p.fields]

            arrays, scalars = [], []
            for v in vals:
                if isinstance(v, (str, bytes)) or np.isscalar(v):
                    scalars.append(v)
                else:
                    arrays.append(np.asarray(v).ravel())

            if arrays:
                params[a] = np.concatenate(arrays, axis=0)
            else:
                if scalars and all(s == scalars[0] for s in scalars):
                    params[a] = scalars[0]
                else:
                    seen = set()
                    uniq = []
                    for s in scalars:
                        if s not in seen:
                            seen.add(s)
                            uniq.append(s)
                    params[a] = uniq

            for p in grp:
                f = p.fields.get(a)
                if f is None:
                    continue
                if f.unit is not None and a not in units_acc:
                    units_acc[a] = f.unit
                if f.label is not None and a not in labels_acc:
                    labels_acc[a] = f.label

        params["units"] = units_acc
        params["labels"] = labels_acc
        out.append(Sample(params))

    return Dataset(out)
