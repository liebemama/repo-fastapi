from copy import deepcopy
from typing import Any


SCHEMA_VERSION = 1


def _jsonable(x: Any) -> Any:
    """
    Convert common data types (NumPy, Torch) to JSON-serializable formats.

    Args:
        x (Any): Input object to be converted.

    Returns:
        Any: JSON-serializable version of the input.
    """
    try:
        import numpy as np
    except Exception:
        np = None

    try:
        import torch
    except Exception:
        torch = None

    if x is None:
        return None
    if np is not None and isinstance(x, np.generic):
        return x.item()
    if np is not None and hasattr(x, "dtype") and hasattr(x, "shape"):
        try:
            return x.tolist()
        except Exception:
            return str(x)
    if torch is not None and hasattr(x, "detach") and hasattr(x, "cpu"):
        try:
            return x.detach().cpu().tolist()
        except Exception:
            return str(x)
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list | tuple | set)):
        return type(x)(_jsonable(v) for v in x)
    if isinstance(x, (str | int | float | bool)):
        return x
    try:
        return str(x)
    except Exception:
        return None


def is_already_unified(raw: dict[str, Any]) -> bool:
    """
    Check if a response is already unified.

    Args:
        raw (Dict[str, Any]): Input dictionary to check.

    Returns:
        bool: True if response is already unified, else False.
    """
    return isinstance(raw, dict) and raw.get("schema_version") is not None and raw.get("status") in ("ok", "error")


def unify_response(provider: str, task: str, raw: Any, request_id: str | None = None) -> dict[str, Any]:
    """
    Standardize responses from different providers with optional metadata and JSON cleaning.

    Args:
        provider (str): Name of the data provider.
        task (str): Task name associated with the response.
        raw (Any): Raw response data.
        request_id (Optional[str]): Optional request identifier.

    Returns:
        Dict[str, Any]: Unified and JSON-serializable response.
    """
    if not isinstance(raw, dict):
        raw = {"result": raw}

    if isinstance(raw, dict) and raw.get("status") in ("ok", "error") and raw.get("schema_version") is None:
        out = deepcopy(raw)
        out.setdefault("provider", provider)
        out.setdefault("task", task)
        out["schema_version"] = SCHEMA_VERSION
        if request_id:
            out.setdefault("meta", {})
            out["meta"]["request_id"] = request_id
        return _jsonable(out)

    if is_already_unified(raw):
        out = deepcopy(raw)
        out.setdefault("provider", provider)
        out.setdefault("task", task)
        if request_id:
            out.setdefault("meta", {})
            out["meta"]["request_id"] = request_id
        return _jsonable(out)

    meta_keys = ("device", "model", "backend", "params", "input", "usage", "input_chars", "truncated_to_1024_tokens")
    meta = {k: raw.get(k) for k in meta_keys if k in raw}

    if "error" in raw:
        err = raw["error"]
        if not isinstance(err, dict):
            err = {"type": "Error", "message": str(err)}
        out = {
            "provider": provider,
            "task": task,
            "status": "error",
            "error": _jsonable(err),
            "schema_version": SCHEMA_VERSION,
        }
        if request_id or meta:
            out["meta"] = _jsonable({**meta, **({"request_id": request_id} if request_id else {})}) or None
        return out

    out = {
        "provider": provider,
        "task": task,
        "status": "ok",
        "elapsed_sec": raw.get("elapsed_sec"),
        "data": _jsonable(raw),
        "schema_version": SCHEMA_VERSION,
    }
    if request_id or meta:
        out["meta"] = _jsonable({**meta, **({"request_id": request_id} if request_id else {})}) or None
    return out
