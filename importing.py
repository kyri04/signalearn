import re
from pathlib import Path
import pandas as pd
import numpy as np
from signalearn.classes import Dataset, Sample
from signalearn.general_utility import snake

_header_re = re.compile(r"^\s*(?P<base>[^\[(]+?)\s*(?:\((?P<u1>[^()]*)\)|\[(?P<u2>[^\[\]]*)\])?\s*$")

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

def _read_table(path, header=True):
    has_header = bool(header)
    header_row = 0 if has_header else None
    df = pd.read_csv(path, sep=None, engine="python", comment="#", dtype="object", header=header_row)
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

def sample_from_file(path, header=True):
    df, labels, units = _read_table(path, header=header)
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
    if "id" not in params:
        params["id"] = path.stem
    return Sample(params)

def _npz_base_stem(path):
    stem = path.stem
    inner = Path(stem)
    if inner.suffix in (".dat", ".txt", ".csv"):
        return inner.stem
    return stem

def sample_from_npz(path):
    with np.load(path, allow_pickle=True) as z:
        params = {}
        labels = {}
        units = {}
        for k in z.files:
            v = z[k]
            if isinstance(v, np.ndarray) and v.shape == ():
                v = v.item()
            params[k] = v
            labels[k] = k
            units[k] = ""

    params["labels"] = labels
    params["units"] = units
    params["name"] = _npz_base_stem(path)
    if "id" not in params:
        params["id"] = params["name"]
    return Sample(params)

def _load_map(map_source):
    df = pd.read_csv(map_source, sep=None, engine="python", dtype="object", index_col=False)
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

def load_data(directory, map=None, format="auto", header=True):
    exts = (".dat", ".txt", ".csv")
    directory = Path(directory)
    map_key = None
    map_data = None
    if map is not None:
        map_key, map_data = _load_map(map)
    map_keys_sorted = sorted(map_data.keys(), key=len, reverse=True) if map_data else None
    samples = []

    if format is None:
        format = "auto"
    fmt = format.lower() if isinstance(format, str) else format

    if fmt == "auto":
        chosen = {}
        for ext in exts:
            for fp in directory.glob(f"*{ext}"):
                chosen.setdefault(fp.stem, fp)
        for fp in directory.glob("*.npz"):
            chosen[_npz_base_stem(fp)] = fp
        files = sorted(chosen.values())
    elif fmt == "npz":
        files = sorted(directory.glob("*.npz"))
    elif fmt in ("raw", "text"):
        files = sorted(fp for ext in exts for fp in directory.glob(f"*{ext}"))
    elif fmt in ("dat", "txt", "csv"):
        files = sorted(directory.glob(f"*.{fmt}"))
    else:
        ext = fmt if isinstance(fmt, str) else ""
        if ext.startswith("."):
            files = sorted(directory.glob(f"*{ext}"))
        else:
            files = sorted(directory.glob(f"*.{ext}"))

    for fp in files:
        sample = sample_from_npz(fp) if fp.suffix == ".npz" else sample_from_file(fp, header=header)
        if map_key and map_data:
            row = None
            if map_key in sample.fields:
                ident = sample.fields[map_key].values
                if isinstance(ident, np.ndarray):
                    if ident.shape == ():
                        ident = ident.item()
                    elif ident.size > 0:
                        ident = ident.ravel()[0]
                row = map_data.get(str(ident))
            else:
                name = sample.name.values if "name" in sample.fields else fp.stem
                if isinstance(name, np.ndarray):
                    if name.shape == ():
                        name = name.item()
                    elif name.size > 0:
                        name = name.ravel()[0]
                name = str(name)
                for key in map_keys_sorted:
                    if key and key in name:
                        row = map_data[key]
                        break
            if row:
                for k, v in row.items():
                    setattr(sample, k, v)
        samples.append(sample)
    return Dataset(samples)
