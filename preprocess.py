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

def trim(points, x_attr, amount, mode='both'):
    mode = mode.lower()

    for point in points:
        x_raw = getattr(point, x_attr, None)
        x = np.asarray(x_raw, dtype=float)

        orig_len = x.shape[0]

        left = x[0]
        right = x[-1]
        if mode in {'front', 'both'}:
            left += amount
        if mode in {'back', 'both'}:
            right -= amount

        mask = (x >= left) & (x <= right)

        attrs_to_trim = []
        for name, value in point.__dict__.items():
            arr = np.asarray(value)
            if arr.ndim >= 1 and arr.shape[0] == orig_len:
                attrs_to_trim.append((name, arr))

        setattr(point, x_attr, x[mask])
        for name, arr in attrs_to_trim:
            setattr(point, name, arr[mask])

    return points

def resample(points, x_axis, rate):
    rate = float(rate)
    step = 1.0 / rate
    eps = np.finfo(float).eps

    for point in points:
        if not hasattr(point, x_axis):
            continue

        x_raw = getattr(point, x_axis)
        x_arr = np.asarray(x_raw, dtype=float).ravel()

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

        updates = {}
        for attr, value in list(point.__dict__.items()):
            if attr == x_axis:
                continue

            arr = np.asarray(value)
            if arr.ndim == 0 or arr.shape[0] != orig_len:
                continue

            try:
                arr_numeric = np.asarray(value, dtype=float)
            except (TypeError, ValueError):
                continue

            arr_valid = arr_numeric[idx]
            interp_func = interp1d(
                x_valid,
                arr_valid,
                axis=0,
                kind='linear',
                bounds_error=False,
                fill_value="extrapolate"
            )
            updates[attr] = interp_func(new_x)

        setattr(point, x_axis, new_x)
        for attr, values in updates.items():
            setattr(point, attr, values)

    return points

def window(points, x_attr, duration, overlap=0.0, allow_partial=False):
    windows = []
    for p_idx, point in enumerate(points):

        x_raw = getattr(point, x_attr)
        x = np.asarray(x_raw, dtype=float).ravel()
        n = x.size

        dx = np.diff(x)
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

            new_params = {}
            for attr, value in point.__dict__.items():
                arr = np.asarray(value)
                if arr.ndim >= 1 and arr.shape[0] == n:
                    try:
                        sliced = value[start:end]
                    except Exception:
                        sliced = arr[start:end]
                    new_params[attr] = sliced
                else:
                    new_params[attr] = value

            parent_id = getattr(point, "name", None)
            new_params["window_index"] = window_idx
            new_params["name"] = f"{parent_id}_w{window_idx}"
            windows.append(point.__class__(new_params))

            window_idx += 1
            start += step

    return windows

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
    feats, means = [], []
    for p in points:
        # y = np.asarray(p.y, float)
        y = getattr(p, y_attr)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        m = y.mean()
        a = trapz(y)
        s = y.std()
        mx = y.max()

        eps=1e-9
        feats.append([np.log(m+eps), np.log(a+eps), s, mx])
        means.append(m)

    X = np.asarray(feats)
    m = np.isfinite(X).all(axis=1)
    X = X[m]
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(Xz)
    post = gmm.predict_proba(Xz)

    empty_comp = np.argmin(gmm.means_[:, 1])
    p_empty = post[:, empty_comp]

    keep_mask = p_empty < tau
    kept = [p for p, k in zip(points, keep_mask) if k]
    dropped = [p for p, k in zip(points, keep_mask) if not k]

    calculate_filtered(points, kept)

    return kept, dropped

def gaussian_mix_saxstemp(points, y_attr: str, tau: float = 0.9):
    """
    Separate 'tissue-like' vs 'background/low-tissue' spectra with a 2-component GMM.

    Features used per spectrum:
      - log(mean intensity)
      - log(integrated intensity)
      - standard deviation
      - max intensity
      - low-q / mid-q log ratio (shape)
      - low-q / high-q log ratio (shape)
      - spectral entropy (shape)

    Parameters
    ----------
    points : iterable
        Objects with attributes:
          - y_attr (e.g. 'intensity' or 'y'): 1D np.ndarray of intensities
          - q : 1D np.ndarray of q positions (same grid for all points)
    y_attr : str
        Name of the attribute on each point that holds the intensity array.
    tau : float, default 0.9
        Posterior probability threshold for being in the *empty/background* component.
        Points with p_empty < tau are kept as 'tissue'; others are dropped.

    Returns
    -------
    kept : list
        Points classified as tissue / high-content.
    dropped : list
        Points classified as empty / low-content.
    """

    eps = 1e-9
    feats = []
    valid_points = []

    # Assume common q-grid
    if not points:
        return [], []

    q = getattr(points[0], "q")
    q = np.asarray(q)
    q_min, q_max = float(q[0]), float(q[-1])
    dq = q_max - q_min

    # Define three broad q-bands for shape features
    q_low  = q_min + 0.33 * dq
    q_mid  = q_min + 0.66 * dq

    low_mask  = q <= q_low
    mid_mask  = (q > q_low) & (q <= q_mid)
    high_mask = q > q_mid

    for p in points:
        y = np.asarray(getattr(p, y_attr), dtype=float)
        if y.ndim != 1 or y.size != q.size:
            continue  # skip malformed spectra

        # clean NaNs/Infs
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- amplitude / power features ----
        m  = y.mean()
        a  = trapz(y, q)                # integrated intensity over q
        s  = y.std()
        mx = y.max()

        # if spectrum is completely zero, treat as background and skip training
        if not np.isfinite(m) or a <= 0.0:
            continue

        # ---- shape features (on normalised spectrum) ----
        y_norm = y / (a + eps)

        low_int  = y_norm[low_mask].sum()
        mid_int  = y_norm[mid_mask].sum()
        high_int = y_norm[high_mask].sum()

        # log ratios of band integrals
        low_mid_ratio  = np.log((low_int  + eps) / (mid_int  + eps))
        low_high_ratio = np.log((low_int  + eps) / (high_int + eps))

        # spectral entropy (shape complexity)
        p_norm = y_norm / (y_norm.sum() + eps)
        entropy = -np.sum(p_norm * np.log(p_norm + eps))

        feat_vec = [
            np.log(m + eps),      # 0: log mean
            np.log(a + eps),      # 1: log area (integrated intensity)
            s,                    # 2: std
            mx,                   # 3: max
            low_mid_ratio,        # 4: low vs mid q
            low_high_ratio,       # 5: low vs high q
            entropy               # 6: spectral entropy
        ]

        if np.all(np.isfinite(feat_vec)):
            feats.append(feat_vec)
            valid_points.append(p)

    if not feats:
        # nothing usable; fall back to keeping everything
        return list(points), []

    X = np.asarray(feats)

    # standardise features
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    # 2-component GMM
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(Xz)
    post = gmm.predict_proba(Xz)

    # identify the "empty/low-content" component:
    # choose component with smallest mean log-area (feature index 1)
    empty_comp = np.argmin(gmm.means_[:, 1])
    p_empty = post[:, empty_comp]

    # keep = tissue-like; drop = empty-like
    keep_mask_valid = p_empty < tau

    kept = [p for p, k in zip(valid_points, keep_mask_valid) if k]
    dropped = [p for p, k in zip(valid_points, keep_mask_valid) if not k]

    # any points that were skipped as invalid we treat as dropped (background)
    skipped_points = [p for p in points if p not in valid_points]
    dropped.extend(skipped_points)

    # update any per-point flags if you rely on them elsewhere
    try:
        calculate_filtered(points, kept)
    except NameError:
        # if you don't have this helper in scope, remove or adapt this call
        pass

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
