import random
import numpy as np
from signalearn.general_utility import *
from signalearn.utility import calculate_filtered
from scipy.fft import *
from scipy.interpolate import interp1d
from scipy.stats import zscore

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from numpy import trapz

def sample(points, f=0.05):
    
    sample_size = int(len(points) * f)
    sampled_points = random.sample(points, sample_size)
    
    return sampled_points

def trim(points, x_attr, y_attr, percent=0.05, front=True, back=True, threshold=1e-8):

    starts, ends = [], []
    for point in points:
        y = getattr(point, y_attr)
        nonzero = np.where(y > threshold)[0]
        if nonzero.size > 0:
            starts.append(nonzero[0])
            ends.append(nonzero[-1] + 1)
    if not starts or not ends:
        return points

    global_start = max(starts)
    global_end   = min(ends)
    usable_len = global_end - global_start

    cut = int(usable_len * percent)
    start = global_start + cut if front else global_start
    end   = global_end - cut if back else global_end

    if end <= start:
        start, end = global_start, global_end

    sl = slice(start, end)

    trimmed = []
    for point in points:
        new_params = point.__dict__.copy()
        new_params["x"] = getattr(point, x_attr)[sl]
        new_params["y"] = getattr(point, y_attr)[sl]
        trimmed.append(point.__class__(new_params))

    return trimmed

def trim(points, x_attr, y_attr, num=3, front=True, back=True, threshold=1e-8):

    starts, ends = [], []
    for point in points:
        y = getattr(point, y_attr)
        nonzero = np.where(y > threshold)[0]
        if nonzero.size > 0:
            starts.append(nonzero[0])
            ends.append(nonzero[-1] + 1)
    if not starts or not ends:
        return points

    global_start = max(starts)
    global_end   = min(ends)
    usable_len = global_end - global_start

    start = global_start + num if front else global_start
    end   = global_end - num if back else global_end

    if end <= start:
        start, end = global_start, global_end

    sl = slice(start, end)

    trimmed = []
    for point in points:
        new_params = point.__dict__.copy()
        new_params["x"] = getattr(point, x_attr)[sl]
        new_params["y"] = getattr(point, y_attr)[sl]
        trimmed.append(point.__class__(new_params))

    return trimmed

def interpolate(points, x_attr, y_attr, n=50):

    interpolated_points = []
    for point in points:
        x = getattr(point, x_attr)
        y = getattr(point, y_attr)

        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 2:
            continue

        f = interp1d(x, y, kind='linear', fill_value="extrapolate")
        x_uniform = np.linspace(x[0], x[-1], n)
        y_uniform = f(x_uniform)

        params = point.__dict__.copy()
        params[x_attr] = x_uniform
        params[y_attr] = y_uniform
        interpolated_points.append(point.__class__(params))

    return interpolated_points

def resample(points, x_attr, rate):
    out = []
    dt = 1.0 / float(rate)

    for p in points:
        t_raw = np.asarray(getattr(p, x_attr), float)
        if t_raw.ndim != 1 or t_raw.size < 2:
            continue

        N = t_raw.shape[0]
        chan_names = []
        for k, v in p.__dict__.items():
            if k == x_attr:
                continue
            if is_numeric_array(v) and np.asarray(v).shape[0] == N:
                chan_names.append(k)
        if not chan_names:
            continue

        t_mask = np.isfinite(t_raw)
        t = t_raw[t_mask]
        if t.size < 2:
            continue
        sidx = np.argsort(t)
        t = t[sidx]
        idx_chain = np.nonzero(t_mask)[0][sidx]
        t_unique, keep = np.unique(t, return_index=True)
        if t_unique.size < 2:
            continue

        t_uniform = np.arange(t_unique[0], t_unique[-1], dt)
        if t_uniform.size < 2:
            t_uniform = np.linspace(t_unique[0], t_unique[-1], 2)

        resampled = {}
        for name in chan_names:
            v_raw = np.asarray(getattr(p, name))
            v = v_raw[idx_chain]
            v = v[keep]

            if v.ndim == 1:
                v = v.astype(float, copy=False)
                v_mask = np.isfinite(v)
                if np.count_nonzero(v_mask) < 2:
                    resampled[name] = getattr(p, name)
                    continue
                f = interp1d(t_unique[v_mask], v[v_mask], bounds_error=False, fill_value="extrapolate", assume_sorted=True)
                resampled[name] = f(t_uniform)
            else:
                v = v.astype(float, copy=False)
                row_finite = np.all(np.isfinite(v), axis=tuple(range(1, v.ndim)))
                if np.count_nonzero(row_finite) < 2:
                    resampled[name] = getattr(p, name)
                    continue
                f = interp1d(t_unique[row_finite], v[row_finite, ...], axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)
                resampled[name] = f(t_uniform)

        params = p.__dict__.copy()
        params[x_attr] = t_uniform
        for name, arr in resampled.items():
            params[name] = arr
        out.append(p.__class__(**params))

    return out

def remove_outliers(points, threshold=3.0, func=np.mean, method='zscore'):
    if not points:
        return []

    values = np.array([func(p.y) for p in points], dtype=float)

    if method == 'zscore':
        zs = zscore(values, nan_policy='omit')
    elif method == 'mad':
        median = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - median))
        if mad == 0:
            return points
        zs = 0.67449 * (values - median) / mad
    else:
        raise ValueError("method must be 'zscore' or 'mad'")

    filtered = [p for p, z in zip(points, zs) if np.abs(z) <= threshold]

    calculate_filtered(points, filtered)

    return filtered

def gaussian_mix(points, y_attr, tau=0.9):
    feats = []
    valid_idx = []

    for i, p in enumerate(points):
        y = np.asarray(getattr(p, y_attr), float)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        if y.size == 0 or not np.isfinite(y).any():
            continue

        m = y.mean()
        a = trapz(y)
        s = y.std()
        mx = y.max()

        eps = 1e-9
        feat = [np.log(m + eps), np.log(a + eps), s, mx]

        if np.all(np.isfinite(feat)):
            feats.append(feat)
            valid_idx.append(i)

    if not feats:
        # nothing usable for GMM; keep everything, drop nothing
        kept = list(points)
        dropped = []
        calculate_filtered(points, kept)
        return kept, dropped

    X = np.asarray(feats)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(Xz)
    post = gmm.predict_proba(Xz)

    # identify "empty" component using inverse-transformed means
    comp_means = scaler.inverse_transform(gmm.means_)
    # index 1 here is log(area); change if you reorder features
    empty_comp = np.argmin(comp_means[:, 1])

    p_empty = post[:, empty_comp]
    keep_mask = p_empty < tau

    kept_idx = [idx for idx, k in zip(valid_idx, keep_mask) if k]
    dropped_idx = [idx for idx, k in zip(valid_idx, keep_mask) if not k]

    # by design, points that were invalid for GMM are kept
    kept = [points[i] for i in range(len(points)) if (i in kept_idx or i not in valid_idx)]
    dropped = [points[i] for i in dropped_idx]

    calculate_filtered(points, kept)
    return kept, dropped

def fourier(points, x_attr, y_attr):
    for point in points:
        xs = getattr(point, x_attr)
        ys = getattr(point, y_attr)

        N = len(ys)

        frequencies = rfftfreq(N, d=(xs[1] - xs[0]))
        amplitudes = rfft(ys) * 2.0 / N 

        setattr(point, x_attr, frequencies)
        setattr(point, y_attr, amplitudes)

    return points

def func_y(points, func=np.mean):

    ys = np.array([p.y for p in points])
    y = func(ys, axis=0)

    return y

def func_points(points, y_attr, func=np.log):

    for point in points:
        setattr(point, y_attr, func(getattr(point, y_attr)))

    return points