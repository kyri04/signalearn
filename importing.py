import re
from pathlib import Path
import pandas as pd
import numpy as np
from signalearn.classes import Dataset, Sample
from signalearn.general_utility import snake

_header_re = re.compile(r"^\s*(?P<base>[^\[(]+?)\s*(?:\((?P<u1>[^()]*)\)|\[(?P<u2>[^\[\]]*)\])?\s*$")

def _is_number(s):
    try:
        float(str(s))
        return True
    except Exception:
        return False

def _parse_header(h):
    m = _header_re.match(str(h))
    if not m:
        name = snake(str(h))
        return name, None, str(h)
    base = m.group("base").strip()
    unit = m.group("u1") or m.group("u2")
    name = snake(base)
    label = base if base else str(h)
    return name, unit.strip() if unit else None, label

def _has_header(path):
    df = pd.read_csv(path, sep=None, engine="python", comment="#", nrows=1, dtype="object")
    cols = list(df.columns)
    if not cols:
        return False
    return not all(_is_number(c) for c in cols)

def _read_table(path):
    has_header = _has_header(path)
    header = 0 if has_header else None
    df = pd.read_csv(path, sep=None, engine="python", comment="#", dtype="object", header=header)
    if not has_header:
        cols = [f"column{i}" for i in range(df.shape[1])]
        df.columns = cols
        labels = {c: c for c in cols}
        units = {c: "" for c in cols}
    else:
        parsed = [_parse_header(c) for c in df.columns]
        cols = [p[0] for p in parsed]
        labels = {p[0]: p[2] for p in parsed}
        units = {p[0]: p[1] or "" for p in parsed}
        df.columns = cols
    return df, labels, units

def sample_from_file(path):
    df, labels, units = _read_table(path)
    params = {}
    for c in df.columns:
        col = df[c]
        if col.nunique(dropna=False) == 1:
            num = pd.to_numeric(col, errors="coerce")
            if num.notna().all():
                val = float(num.iloc[0])
            else:
                val = col.iloc[0]
            params[c] = val
        else:
            num = pd.to_numeric(col, errors="coerce")
            if num.notna().all():
                params[c] = num.to_numpy(dtype=float)
            else:
                params[c] = col.to_numpy()
    params["labels"] = labels
    params["units"] = units
    params["name"] = path.stem
    return Sample(params)

def _load_map(map_source):
    df = pd.read_csv(map_source, sep=None, engine="python", dtype="object")
    cols = list(df.columns)
    if not cols:
        return None, None
    key_col = cols[0]
    conv = {}
    for c in cols[1:]:
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().all():
            conv[c] = num
        else:
            conv[c] = df[c]
    data = {}
    keys = df[key_col].astype(str)
    for i, k in enumerate(keys):
        row = {c: conv[c].iloc[i] for c in cols[1:]}
        data[k] = row
    return key_col, data

def load_data(directory, map=None):
    exts = (".dat", ".txt", ".csv")
    directory = Path(directory)
    map_key = None
    map_data = None
    if map is not None:
        map_key, map_data = _load_map(map)
    samples = []
    for ext in exts:
        for fp in sorted(directory.glob(f"*{ext}")):
            sample = sample_from_file(fp)
            if map_key and map_data and map_key in sample.fields:
                ident = sample.fields[map_key].values
                if isinstance(ident, np.ndarray):
                    if ident.shape == ():
                        ident = ident.item()
                    elif ident.size > 0:
                        ident = ident.ravel()[0]
                row = map_data.get(str(ident))
                if row:
                    for k, v in row.items():
                        setattr(sample, k, v)
            samples.append(sample)
    return Dataset(samples)
