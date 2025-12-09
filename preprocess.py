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

def func_points(points, y_attr, func=np.log):

    for point in points:
        setattr(point, y_attr, func(getattr(point, y_attr)))

    return points

def quantify_peak(points, x_attr, y_attr, center=None, window=None, suffix=None,
                  sigma_factor=3.0, min_width=None):

    if isinstance(y_attr, str):
        y_attrs = [y_attr]
    elif isinstance(y_attr, (list, tuple)):
        y_attrs = list(y_attr)

    use_window = center is not None and window is not None

    for point in points:
        if not hasattr(point, x_attr):
            continue

        x_raw = getattr(point, x_attr)
        x = np.asarray(x_raw, dtype=float).ravel()
        if x.size < 2 or not np.any(np.isfinite(x)):
            continue

        units = getattr(point, "units", None)
        labels = getattr(point, "labels", None)
        x_unit = None
        if isinstance(units, dict):
            x_unit = units.get(x_attr, None)

        for attr in y_attrs:
            if not hasattr(point, attr):
                continue

            peak_x_val = np.nan
            fwhm_val = np.nan

            y_raw = getattr(point, attr)
            y = np.asarray(y_raw, dtype=float).ravel()

            if y.size != x.size:
                peak_name = f"{attr}{'_' + suffix or ''}_peak_pos"
                fwhm_name = f"{attr}{'_' + suffix or ''}_fwhm"
                setattr(point, peak_name, peak_x_val)
                setattr(point, fwhm_name, fwhm_val)
                continue

            mask = np.isfinite(x) & np.isfinite(y)
            x_valid = x[mask]
            y_valid = y[mask]

            if x_valid.size < 2:
                peak_name = f"{attr}{'_' + suffix or ''}_peak_pos"
                fwhm_name = f"{attr}{'_' + suffix or ''}_fwhm"
                setattr(point, peak_name, peak_x_val)
                setattr(point, fwhm_name, fwhm_val)
                continue

            if use_window:
                win_mask = (x_valid >= center - window) & (x_valid <= center + window)
                x_use = x_valid[win_mask]
                y_use = y_valid[win_mask]
            else:
                x_use = x_valid
                y_use = y_valid

            if x_use.size < 2:
                peak_name = f"{attr}{'_' + suffix or ''}_peak_pos"
                fwhm_name = f"{attr}{'_' + suffix or ''}_fwhm"
                setattr(point, peak_name, peak_x_val)
                setattr(point, fwhm_name, fwhm_val)
                continue

            # Estimate local noise level (MAD-based sigma) for optional prominence threshold
            y_med = float(np.nanmedian(y_use))
            r = y_use - y_med
            mad = float(np.nanmedian(np.abs(r)))
            sigma = 1.4826 * mad

            min_prominence = None
            if sigma_factor is not None and sigma_factor > 0 and np.isfinite(sigma) and sigma > 0.0:
                min_prominence = sigma_factor * sigma

            # Use scipy.signal.find_peaks with optional prominence/width thresholds
            if min_prominence is not None:
                peaks, props = find_peaks(y_use, prominence=min_prominence, width=min_width)
            else:
                peaks, props = find_peaks(y_use, width=min_width)

            if peaks.size == 0:
                peak_name = f"{attr}{'_' + suffix or ''}_peak_pos"
                fwhm_name = f"{attr}{'_' + suffix or ''}_fwhm"
                setattr(point, peak_name, peak_x_val)
                setattr(point, fwhm_name, fwhm_val)
                continue

            # Choose best peak index
            if use_window and center is not None:
                # pick the peak whose x position is closest to the expected center
                peak_xs = x_use[peaks]
                best_idx = int(np.argmin(np.abs(peak_xs - center)))
            else:
                # pick the highest peak
                best_idx = int(np.argmax(y_use[peaks]))

            peak_sample_index = peaks[best_idx]
            peak_x = float(x_use[peak_sample_index])

            # Compute FWHM using peak_widths at half prominence
            try:
                widths, height, left_ips, right_ips = peak_widths(
                    y_use, peaks=[peak_sample_index], rel_height=0.5
                )
            except Exception:
                widths = [np.nan]
                left_ips = [np.nan]
                right_ips = [np.nan]

            w_samples = float(widths[0])
            li = float(left_ips[0])
            ri = float(right_ips[0])

            if not (np.isfinite(w_samples) and np.isfinite(li) and np.isfinite(ri)):
                peak_name = f"{attr}{'_' + suffix or ''}_peak_pos"
                fwhm_name = f"{attr}{'_' + suffix or ''}_fwhm"
                setattr(point, peak_name, peak_x_val)
                setattr(point, fwhm_name, fwhm_val)
                continue

            # Convert sample indices (which may be fractional) to x positions
            idx_array = np.arange(x_use.size, dtype=float)
            left_x = float(np.interp(li, idx_array, x_use))
            right_x = float(np.interp(ri, idx_array, x_use))

            if np.isfinite(left_x) and np.isfinite(right_x) and right_x > left_x:
                peak_x_val = peak_x
                fwhm_val = float(right_x - left_x)

            base_name = f"{attr}{'_' + suffix or ''}"
            peak_name = f"{base_name}_peak_pos"
            fwhm_name = f"{base_name}_fwhm"

            setattr(point, peak_name, peak_x_val)
            setattr(point, fwhm_name, fwhm_val)

            if isinstance(units, dict):
                units[peak_name] = x_unit
                units[fwhm_name] = x_unit

            if isinstance(labels, dict):
                base_label = labels.get(attr, attr)
                labels[peak_name] = f"{base_label} peak position"
                labels[fwhm_name] = f"{base_label} FWHM"

    return points

def quantify_peaks(points, x_attr, y_attr, centers, window, sigma_factor=3.0, min_width=None):

    for idx, c in enumerate(centers, start=1):
        suffix = f"_p{idx}"
        quantify_peak(points, x_attr, y_attr,
                      center=c, window=window, suffix=suffix,
                      sigma_factor=sigma_factor, min_width=min_width)

    return points
