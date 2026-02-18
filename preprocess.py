import random
import numpy as np
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from signalearn.classes import Dataset, Sample
from signalearn.general_utility import pretty_func
from signalearn.utility import new_sample

def sample(dataset, f=0.05):
    sample_size = int(len(dataset) * f)
    sampled = random.sample(dataset.samples, sample_size)
    return Dataset([new_sample(s) for s in sampled])

def trim(x, amount, mode='both'):
    mode = mode.lower()
    dataset = x._dataset
    samples = []
    x_name = x.name
    for sample in dataset.samples:
        x_field = sample.fields.get(x_name)
        if x_field is None:
            continue
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
        labels = {}
        units = {}
        for name, field in sample.fields.items():
            arr = np.asarray(field.values)
            if arr.ndim >= 1 and arr.shape[0] == orig_len:
                updates[name] = arr[mask]
                labels[name] = field.label
                units[name] = field.unit
        params = {"id": sample.fields["id"].values, **updates, "labels": labels, "units": units}
        samples.append(Sample(params))
    return Dataset(samples)

def select(x, start, end):
    dataset = x._dataset
    samples = []
    x_name = x.name
    for sample in dataset.samples:
        x_field = sample.fields.get(x_name)
        if x_field is None:
            continue
        x_vals = np.asarray(x_field.values, dtype=float)
        orig_len = x_vals.shape[0]
        mask = (x_vals >= start) & (x_vals <= end)
        updates = {}
        labels = {}
        units = {}
        for name, field in sample.fields.items():
            arr = np.asarray(field.values)
            if arr.ndim >= 1 and arr.shape[0] == orig_len:
                updates[name] = arr[mask]
                labels[name] = field.label
                units[name] = field.unit
        params = {"id": sample.fields["id"].values, **updates, "labels": labels, "units": units}
        samples.append(Sample(params))
    return Dataset(samples)

def resample(x, rate=None, n=None):
    if n is not None:
        n = int(n)
    else:
        rate = float(rate)
        step = 1.0 / rate
    eps = np.finfo(float).eps
    dataset = x._dataset
    x_name = x.name
    samples = []
    for sample in dataset.samples:
        x_field = sample.fields.get(x_name)
        if x_field is None:
            continue
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

        if n is not None:
            if n < 2:
                n = 2
            new_x = np.linspace(x_valid[0], x_valid[-1], n)
        else:
            span = x_valid[-1] - x_valid[0]
            n_samples = max(2, int(np.floor(span * rate)) + 1)
            new_x = x_valid[0] + np.arange(n_samples) * step
            if new_x[-1] < x_valid[-1] - eps:
                new_x = np.append(new_x, x_valid[-1])
        updates = {x_name: new_x}
        labels = {x_name: x_field.label}
        units = {x_name: x_field.unit}
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
            labels[attr] = field.label
            units[attr] = field.unit
        params = {"id": sample.fields["id"].values, **updates, "labels": labels, "units": units}
        samples.append(Sample(params))
    return Dataset(samples)

def window(x, duration, overlap=0.0, allow_partial=False):
    dataset = x._dataset
    x_name = x.name
    windows = []
    for sample in dataset.samples:
        x_field = sample.fields.get(x_name)
        if x_field is None:
            continue
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
            labels = {}
            units = {}
            for attr, field in sample.fields.items():
                arr = np.asarray(field.values)
                if arr.ndim >= 1 and arr.shape[0] == n:
                    updates[attr] = arr[start:end]
                    labels[attr] = field.label
                    units[attr] = field.unit
            updates["window_index"] = window_idx
            labels["window_index"] = "Window Index"
            units["window_index"] = ""
            params = {"id": sample.fields["id"].values, **updates, "labels": labels, "units": units}
            windows.append(Sample(params))
            window_idx += 1
            start += step
    return Dataset(windows)

def func(y, func=np.log):
    dataset = y._dataset
    y_name = y.name
    out_name = y_name
    samples = []
    for sample in dataset.samples:
        y_field = sample.fields.get(y_name)
        if y_field is None:
            continue
        params = {"id": sample.fields["id"].values, out_name: func(y_field.values)}
        params["labels"] = {out_name: pretty_func(y_field.label or y_name, func)}
        params["units"] = {out_name: y_field.unit}
        samples.append(Sample(params))
    return Dataset(samples)

def normalise(y):
    dataset = y._dataset
    y_name = y.name
    out_name = y_name
    samples = []
    for sample in dataset.samples:
        y_field = sample.fields.get(y_name)
        if y_field is None:
            continue
        y_vals = np.asarray(y_field.values, dtype=float)
        total = float(np.sum(y_vals))
        if total <= 0 or not np.isfinite(total):
            continue
        params = {"id": sample.fields["id"].values, out_name: y_vals / total}
        params["labels"] = {out_name: f"Normalised {y_field.label or y_name}"}
        params["units"] = {out_name: ""}
        samples.append(Sample(params))
    return Dataset(samples)

def baseline(y, lam=1e5, p=0.01, niter=10):
    dataset = y._dataset
    y_name = y.name
    out_name = y_name
    cache = {}
    samples = []
    for sample in dataset.samples:
        y_field = sample.fields.get(y_name)
        if y_field is None:
            continue
        y_vals = np.asarray(y_field.values, dtype=float).ravel()
        n = int(y_vals.size)
        if n < 3:
            z = y_vals.copy()
            params = {"id": sample.fields["id"].values, out_name: z}
            params["labels"] = {out_name: f"{(y_field.label or y_name)} Baseline"}
            params["units"] = {out_name: y_field.unit}
            samples.append(Sample(params))
            continue
        if n not in cache:
            D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
            cache[n] = D.T @ D
        DTD = cache[n]
        w = np.ones(n, dtype=float)
        for _ in range(int(niter)):
            W = sparse.spdiags(w, 0, n, n)
            Z = W + float(lam) * DTD
            z = spsolve(Z, w * y_vals)
            w = float(p) * (y_vals > z) + (1.0 - float(p)) * (y_vals <= z)

        params = {"id": sample.fields["id"].values, out_name: z}
        params["labels"] = {out_name: f"{(y_field.label or y_name)} Baseline"}
        params["units"] = {out_name: y_field.unit}
        samples.append(Sample(params))
    return Dataset(samples)

def peaks(x, y, min_peak_distance=0.004, prominence_frac=0.03, match_window=0.003):
    dataset = y._dataset
    x_dataset = x._dataset
    x_name = x.name
    y_name = y.name

    x_by_id = {}
    for sample in x_dataset.samples:
        if "id" not in sample.fields or x_name not in sample.fields:
            continue
        x_by_id[str(sample.fields["id"].values)] = np.asarray(sample.fields[x_name].values, dtype=float).ravel()

    aligned = []
    for sample in dataset.samples:
        sid = str(sample.fields["id"].values)
        if sid in x_by_id and y_name in sample.fields:
            aligned.append(
                (sid, x_by_id[sid], np.asarray(sample.fields[y_name].values, dtype=float).ravel())
            )

    x_vals = aligned[0][1]
    y_all = np.asarray([v[2] for v in aligned], dtype=float)
    y_median = np.median(y_all, axis=0)

    win = min(11, len(y_median) - (1 - len(y_median) % 2))
    if win < 5:
        y_for_peaks = y_median
    else:
        y_for_peaks = savgol_filter(y_median, window_length=win, polyorder=2)

    prominence_min = float(prominence_frac) * (float(np.max(y_for_peaks)) - float(np.min(y_for_peaks)))
    dq = float(np.median(np.diff(x_vals)))
    min_peak_distance_pts = max(1, int(round(float(min_peak_distance) / max(dq, 1e-12))))
    ref_peaks, _ = find_peaks(
        y_for_peaks,
        distance=min_peak_distance_pts,
        prominence=prominence_min,
    )
    ref_peak_x = x_vals[ref_peaks]
    x_unit = x.fields[0].unit or ""
    y_unit = y.fields[0].unit or ""

    samples = []
    for sample in dataset.samples:
        sid = sample.fields["id"].values
        sid_key = str(sid)
        if sid_key not in x_by_id:
            continue
        x_i = x_by_id[sid_key]
        y_i = np.asarray(sample.fields[y_name].values, dtype=float).ravel()

        if win < 5:
            ys = y_i
        else:
            w = min(win, len(y_i) - (1 - len(y_i) % 2))
            ys = y_i if w < 5 else savgol_filter(y_i, window_length=w, polyorder=2)

        dq_i = float(np.median(np.diff(x_i)))
        min_dist_pts_i = max(1, int(round(float(min_peak_distance) / max(dq_i, 1e-12))))
        prom_min_i = float(prominence_frac) * (float(np.max(ys)) - float(np.min(ys)))
        peaks_i, props_i = find_peaks(ys, distance=min_dist_pts_i, prominence=prom_min_i)

        peak_x_i = np.array([], dtype=float)
        fwhm_x_i = np.array([], dtype=float)
        prom_i = np.array([], dtype=float)
        if peaks_i.size > 0:
            _, _, left_ips_i, right_ips_i = peak_widths(ys, peaks_i, rel_height=0.5)
            idx_i = np.arange(len(x_i), dtype=float)
            left_x_i = np.interp(left_ips_i, idx_i, x_i)
            right_x_i = np.interp(right_ips_i, idx_i, x_i)
            fwhm_x_i = right_x_i - left_x_i
            peak_x_i = x_i[peaks_i]
            prom_i = props_i["prominences"]

        params = {"id": sid}
        labels = {}
        units = {}
        for k, x_ref in enumerate(ref_peak_x, start=1):
            pos_name = f"pos{k}"
            fwhm_name = f"fwhm{k}"
            prom_name = f"prominence{k}"
            params[pos_name] = np.nan
            params[fwhm_name] = np.nan
            params[prom_name] = np.nan
            if peaks_i.size > 0:
                d = np.abs(peak_x_i - x_ref)
                j = int(np.argmin(d))
                if d[j] <= float(match_window):
                    params[pos_name] = float(peak_x_i[j])
                    params[fwhm_name] = float(fwhm_x_i[j])
                    params[prom_name] = float(prom_i[j])
            labels[pos_name] = f"Peak {k} Position"
            labels[fwhm_name] = f"Peak {k} FWHM"
            labels[prom_name] = f"Peak {k} Prominence"
            units[pos_name] = x_unit
            units[fwhm_name] = x_unit
            units[prom_name] = y_unit

        params["labels"] = labels
        params["units"] = units
        samples.append(Sample(params))

    return Dataset(samples)

def subtract(a, b):
    out_name = a.name
    a_ds = a._dataset
    b_ds = b._dataset
    b_map = {}
    for s in b_ds.samples:
        if b.name not in s.fields:
            continue
        b_map[str(s.fields["id"].values)] = s.fields[b.name].values

    samples = []
    for s in a_ds.samples:
        a_field = s.fields.get(out_name)
        if a_field is None:
            continue
        key = str(s.fields["id"].values)
        if key not in b_map:
            continue
        res = np.asarray(a_field.values, dtype=float) - np.asarray(b_map[key], dtype=float)
        params = {"id": s.fields["id"].values, out_name: res}
        a_label = a_field.label or out_name
        b_label = b.label or b.name
        params["labels"] = {out_name: f"{a_label} - {b_label}"}
        params["units"] = {out_name: a_field.unit}
        samples.append(Sample(params))
    return Dataset(samples)

def filter(x, val, includes=True):
    dataset = x._dataset
    name = x.name
    if isinstance(val, (list, tuple, set, np.ndarray)):
        vals = list(val)
    else:
        vals = [val]

    samples = []
    for sample in dataset.samples:
        field = sample.fields.get(name)
        attr_val = field.values if field is not None else None
        matched = False
        for v in vals:
            if isinstance(v, str):
                if attr_val is None:
                    continue
                matched = str(attr_val) == v
            else:
                matched = attr_val == v
            if matched:
                break
        if matched == includes:
            samples.append(new_sample(sample))
    return Dataset(samples)

def between(x, min=None, max=None):
    dataset = x._dataset
    samples = []
    for sample, field in zip(dataset.samples, x.fields):
        v = field.values
        if v is None or isinstance(v, (str, bytes)) or isinstance(v, (list, tuple, np.ndarray)):
            continue
        v = float(v)
        if min is not None and v < float(min):
            continue
        if max is not None and v > float(max):
            continue
        samples.append(new_sample(sample))
    return Dataset(samples)

def where(x, condition):
    dataset = x._dataset
    samples = []
    for sample, field in zip(dataset.samples, x.fields):
        if condition(np.asarray(field.values)):
            samples.append(new_sample(sample))
    return Dataset(samples)

def concat(datasets):
    samples = []
    for i, dataset in enumerate(datasets):
        src = f"dataset{i}"
        for sample in dataset.samples:
            samples.append(new_sample(sample, {"source": src}))
    return Dataset(samples)

def take(dataset, ids):
    if hasattr(ids, "fields"):
        wanted = {str(f.values) for f in ids.fields}
    elif isinstance(ids, (list, tuple, set, np.ndarray)):
        wanted = {str(v) for v in ids}
    else:
        wanted = {str(ids)}

    samples = []
    for sample in dataset.samples:
        if "id" not in sample.fields:
            continue
        if str(sample.fields["id"].values) in wanted:
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

        params = {"id": str(k), match_name: k}
        name_vals = []
        for p in grp:
            if "name" in p.fields:
                name_vals.append(str(p.fields["name"].values))
        params["name"] = "+".join(sorted(set(name_vals))) if name_vals else str(k)

        meta = {"id", "name", match_name}
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
