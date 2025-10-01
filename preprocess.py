import random
import numpy as np
from signalearn.general_utility import *
from scipy.fft import *
from scipy.interpolate import interp1d
from scipy.stats import zscore

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

def remove_outliers(points, threshold=3.0, func=np.mean):

    if not points:
        return []

    values = np.array([func(p.y) for p in points], dtype=float)
    zs = zscore(values, nan_policy='omit')
    filtered = [p for p, z in zip(points, zs) if np.abs(z) <= threshold]

    original_count = len(points)
    removed_count = original_count - len(filtered)
    removed_percentage = (removed_count / original_count) * 100 if original_count > 0 else 0
    print(f"Removed {removed_percentage:.2f}% of the data.")

    return filtered

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
        x_uniform = np.linspace(x[0], x[-1], n)
        y_uniform = f(x_uniform)

        params = point.__dict__.copy()
        params["x"] = x_uniform
        params["y"] = y_uniform
        interpolated_points.append(point.__class__(params))

    return interpolated_points

def fourier(points):

    for point in points:
        N = len(point.y)

        frequencies = rfftfreq(N, d=(point.x[1] - point.x[0]))
        # amplitudes = rfft(invert(point.y)) * 2.0 / N 
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