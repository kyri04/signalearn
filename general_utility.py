import os
from datetime import datetime
import gc
import re
import inspect, ast, textwrap
from numbers import Number
import numpy as np
import subprocess
from pathlib import Path
from types import SimpleNamespace

def name_from_path(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]

def invert(y):
    for i in range(len(y)):
        y[i] = - y[i]

    return y

def snake(s: str):
    s = s.strip()
    s = re.sub(r"[^0-9A-Za-z]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_").lower()

def remove_trailing_letters(s):
    
    end_index = len(s)
    for i in range(len(s) - 1, -1, -1):
        if s[i].isdigit():
            end_index = i + 1
            break

    return s[:end_index]

def time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 

def cleanup():
    gc.collect()

def get_attributes(cls):
    src = inspect.getsource(cls.__init__)
    src = textwrap.dedent(src)
    tree = ast.parse(src)
    return [
        node.targets[0].attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Attribute)
        and isinstance(node.targets[0].value, ast.Name)
        and node.targets[0].value.id == "self"
    ]

def to_float(x):
        if x is None:
            return None
        try:
            if isinstance(x, np.ndarray):
                if x.size == 0:
                    return None
                x = x.ravel()[0]
            if not isinstance(x, Number):
                x = float(x)
            if isinstance(x, float) and np.isnan(x):
                return None
            return float(x)
        except Exception:
            return None

def pct(x):
        v = to_float(x)
        return "N/A" if v is None else f"{v*100:.2f}%"

def num(x):
    v = to_float(x)
    return "N/A" if v is None else f"{v:.2f}"

def is_number(x):
        return isinstance(x, (int, float, np.integer, np.floating))

def is_numeric_array(a):
    try:
        arr = np.asarray(a)
    except Exception:
        return False
    return arr.ndim >= 1 and np.issubdtype(arr.dtype, np.number)

def format_confusion_matrix(conf_matrix, labels):
        col_width = max(5, max(len(str(label)) for label in labels))
        row_header_width = max(len("Actual " + str(label)) for label in labels)
        output = []
        header_width = len(labels) * (col_width + 1) - 1
        output.append(" " * (row_header_width + 1) + "Predicted".center(header_width))
        output.append(" " * (row_header_width + 1) + " ".join(f"{label:^{col_width}}" for label in labels))
        for i, label in enumerate(labels):
            row_header = f"Actual {label}"
            row_values = " ".join(f"{conf_matrix[i, j]:^{col_width}}" for j in range(len(labels)))
            output.append(f"{row_header:<{row_header_width}} " + row_values)
        return "\n".join(output)

def format_attributes(obj, title):
    lines = [title]
    for k, v in vars(obj).items():
        if isinstance(v, float):
            v = f"{v:.3f}"
        else:
            v = str(v)

        if "\n" in v:
            lines.append(f"{k}:\n{v}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)

def pretty(s):
    s = str(s).strip()
    parts = s.replace("-", "_").split("_")
    parts = [p for p in parts if p]
    return " ".join(p.capitalize() for p in parts) if parts else ""

def pretty_func(label, func):
    if not callable(func):
        return label
    name = pretty(func.__name__)
    return f"{name} ({label})" if name else label

def display_metadata(markdown: bool = False) -> str:

    def _repo_root() -> Path:
        here = Path(__file__).resolve()
        for parent in [here] + list(here.parents):
            if (parent / ".git").exists():
                return parent
        raise RuntimeError("Could not locate git repository root")

    def _git(args: list[str]) -> str:
        repo = _repo_root()
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
        ).strip()

    repo = _repo_root()

    commit = _git(["rev-parse", "--short", "HEAD"])
    full_commit = _git(["rev-parse", "HEAD"])
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])

    dirty = (
        subprocess.call(["git", "-C", str(repo), "diff", "--quiet"]) != 0
        or subprocess.call(["git", "-C", str(repo), "diff", "--cached", "--quiet"]) != 0
    )

    status = "DIRTY" if dirty else "CLEAN"

    if markdown:
        return (
            f"`signalearn` · **branch** `{branch}` · "
            f"**commit** `{commit}` · **{status}**  \n"
            f"`{full_commit}`"
        )

    return (
        f"signalearn | branch {branch} | commit {commit} | "
        f"{status} | {full_commit}"
    )

def make_namespace(d):
    return SimpleNamespace(**d)