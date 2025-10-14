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

def trim(points, percent=0.05, front=True, back=True, threshold=1e-8):

    starts, ends = [], []
    for point in points:
        y = np.array(point.y)
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
        new_params["x"] = np.array(point.x)[sl]
        new_params["y"] = np.array(point.y)[sl]
        trimmed.append(point.__class__(new_params))

    return trimmed

def trim(points, num=3, front=True, back=True, threshold=1e-8):

    starts, ends = [], []
    for point in points:
        y = np.array(point.y)
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
        new_params["x"] = np.array(point.x)[sl]
        new_params["y"] = np.array(point.y)[sl]
        trimmed.append(point.__class__(new_params))

    return trimmed

def interpolate(points, n=50):

    interpolated_points = []
    for point in points:
        x = np.array(point.x, dtype=float)
        y = np.array(point.y, dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 2:
            continue

        f = interp1d(x, y, kind='linear', fill_value="extrapolate")
        x_uniform = np.linspace(x[0], y[-1], n)
        y_uniform = f(x_uniform)

        params = point.__dict__.copy()
        params["x"] = x_uniform
        params["y"] = y_uniform
        interpolated_points.append(point.__class__(params))

    return interpolated_points

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

def gaussian_mix(points, tau=0.9, n_components=2):
    feats, means = [], []
    for p in points:
        y = np.asarray(p.y, float)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        m = y.mean()
        a = trapz(y)
        s = y.std()
        mx = y.max()

        eps=1e-9
        feats.append([np.log(m+eps), np.log(a+eps), s, mx])
        means.append(m)

    X = np.asarray(feats)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(Xz)
    post = gmm.predict_proba(Xz)

    empty_comp = np.argmin(gmm.means_[:, 1])
    p_empty = post[:, empty_comp]

    keep_mask = p_empty < tau
    kept = [p for p, k in zip(points, keep_mask) if k]
    dropped = [p for p, k in zip(points, keep_mask) if not k]

    calculate_filtered(points, kept)

    return kept, dropped

def fourier(points):

    for point in points:
        N = len(point.y)

        frequencies = rfftfreq(N, d=(point.x[1] - point.x[0]))
        amplitudes = rfft(point.y) * 2.0 / N 

        point.x = frequencies
        point.y = amplitudes

    return points

def func_y(points, func=np.mean):

    ys = np.array([p.y for p in points])
    y = func(ys, axis=0)

    return y

def func_points(points, func=np.log):

    for point in points:
        point.y = func(point.y)

    return points