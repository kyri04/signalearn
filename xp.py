from __future__ import annotations
import os
import warnings
from typing import Tuple, Any, Callable

__all__ = [
    "xp", "sp", "BACKEND", "is_gpu", "to_cpu", "to_gpu", "asnumpy", "ascupy",
    "cpu_fallback", "use_backend", "get_backend"
]

_BACKEND_ENV = os.getenv("SIGNALEARN_XP", "auto").lower()  # "auto" | "cupy" | "numpy"

def _try_cupy() -> Tuple[Any, Any, str]:
    import importlib
    cupy = importlib.import_module("cupy")
    # smoke test: ensure we actually have a working CUDA context
    _ = cupy.zeros((1,), dtype=cupy.float32)
    try:
        csp = importlib.import_module("cupyx.scipy")
    except Exception:
        csp = None
        warnings.warn("cupyx.scipy not available; SciPy calls will need cpu_fallback.", RuntimeWarning)
    return cupy, csp, "cupy"

def _numpy_backend() -> Tuple[Any, Any, str]:
    import numpy as np
    import scipy as sp
    return np, sp, "numpy"

def _select_backend(prefer_gpu: bool = True) -> Tuple[Any, Any, str]:
    if _BACKEND_ENV == "cupy" or (_BACKEND_ENV == "auto" and prefer_gpu):
        try:
            return _try_cupy()
        except Exception as e:
            if _BACKEND_ENV == "cupy":
                raise RuntimeError(
                    "SIGNALEARN_XP=cupy was requested but CuPy/CUDA isn't available."
                ) from e
            warnings.warn("Falling back to NumPy/SciPy: CuPy not available/usable.", RuntimeWarning)
    return _numpy_backend()

# Initialize default backend at import time
xp, sp, BACKEND = _select_backend(prefer_gpu=True)

def is_gpu() -> bool:
    return BACKEND == "cupy"

def asnumpy(a):
    """Always return a NumPy array on host."""
    if is_gpu():
        import cupy as cp
        return cp.asnumpy(a)
    import numpy as np
    return np.asarray(a)

def ascupy(a):
    """Return a CuPy array; if backend is CPU, raises unless overridden by `use_backend("cupy")`."""
    if not is_gpu():
        raise RuntimeError("ascupy() called but GPU backend is not active.")
    import cupy as cp
    return cp.asarray(a)

def to_cpu(*arrays):
    """Map arrays to host NumPy."""
    return tuple(asnumpy(a) for a in arrays)

def to_gpu(*arrays):
    """Map arrays to device CuPy (only when GPU backend active)."""
    return tuple(ascupy(a) for a in arrays)

def cpu_fallback(fn: Callable) -> Callable:
    """
    Decorator: when running on GPU but the function isn't implemented in cupyx.scipy,
    run it on CPU (NumPy/SciPy) by copying inputs to host, then copy result back to GPU.
    """
    def wrapped(*args, **kwargs):
        if not is_gpu():
            return fn(*args, **kwargs)
        try:
            return fn(*args, **kwargs)
        except (NotImplementedError, AttributeError):
            # Move everything to CPU, call the NumPy/SciPy equivalent, send back if array-like
            host_args = tuple(asnumpy(a) if _looks_like_array(a) else a for a in args)
            host_kwargs = {k: (asnumpy(v) if _looks_like_array(v) else v) for k, v in kwargs.items()}
            import numpy as _np
            res = fn.__wrapped_cpu__(*host_args, **host_kwargs)  # see pairing below
            # Map result back to GPU
            import cupy as cp
            return _map_result(lambda x: cp.asarray(x), res)
    return wrapped

def _looks_like_array(x):
    return hasattr(x, "__array__") or hasattr(x, "__cuda_array_interface__") or hasattr(x, "__array_priority__")

def _map_result(mapper, res):
    # Map nested structures (tuple/list/dict) while preserving shapes
    from collections.abc import Mapping, Sequence
    if isinstance(res, Mapping):
        return res.__class__({k: _map_result(mapper, v) for k, v in res.items()})
    if isinstance(res, (tuple, list)):
        return res.__class__(_map_result(mapper, v) for v in res)
    # single object
    try:
        return mapper(res)
    except Exception:
        return res  # non-array scalars/objects

def use_backend(name: str):
    """
    Context manager to temporarily switch backend.
    Example:
        with use_backend("numpy"):
            ...
    """
    from contextlib import contextmanager
    import types

    @contextmanager
    def _ctx():
        global xp, sp, BACKEND
        old = (xp, sp, BACKEND)
        if name.lower() == "cupy":
            xp, sp, BACKEND = _try_cupy()
        elif name.lower() == "numpy":
            xp, sp, BACKEND = _numpy_backend()
        else:
            raise ValueError("use_backend: expected 'numpy' or 'cupy'")
        try:
            yield
        finally:
            xp, sp, BACKEND = old
    return _ctx()

def get_backend() -> str:
    return BACKEND
