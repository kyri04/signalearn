import random
import numpy as np
from signalearn.general_utility import *
from signalearn.utility import calculate_filtered
from scipy.fft import *
from scipy.interpolate import interp1d
from scipy.stats import zscore
from scipy.signal import find_peaks, peak_widths
import numpy as np

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

def select(points, x_attr, start, end):
    for point in points:
        x_raw = getattr(point, x_attr, None)
        x = np.asarray(x_raw, dtype=float)

        orig_len = x.shape[0]

        mask = (x >= start) & (x <= end)

        attrs_to_select = []
        for name, value in point.__dict__.items():
            arr = np.asarray(value)
            if arr.ndim >= 1 and arr.shape[0] == orig_len:
                attrs_to_select.append((name, arr))

        setattr(point, x_attr, x[mask])
        for name, arr in attrs_to_select:
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

def func(points, y_attr, func=np.log):

    for point in points:
        setattr(point, y_attr, func(getattr(point, y_attr)))

    return points
