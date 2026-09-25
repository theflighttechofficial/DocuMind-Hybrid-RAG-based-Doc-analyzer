"""Thread-safe, load-once registry for heavy models.

Background processing and request handlers run in different threads. Two
threads lazily constructing the same Hugging Face model at the same time can
leave torch in a broken state ("Cannot copy out of meta tensor; no data!"),
which then poisons every later load in the process. All model loading goes
through load_model() so construction is serialized and cached, and so a
failed primary model falls back to its lighter alternative instead of
failing the request.
"""
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_models: dict[str, object] = {}
_info: dict[str, dict] = {}


def _rss_mb() -> float | None:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2**20
    except Exception:
        return None


def load_model(key: str, candidates: list[tuple[str, callable]]):
    """candidates: [(model_name, factory), ...] tried in order until one loads.
    Returns the loaded object; raises the last error if every candidate fails.
    """
    if key in _models:
        return _models[key]
    with _lock:
        if key in _models:
            return _models[key]
        last_exc = None
        for i, (name, factory) in enumerate(candidates):
            before, t0 = _rss_mb(), time.time()
            try:
                model = factory()
            except Exception as exc:  # download failure, OOM, missing package
                last_exc = exc
                logger.warning("Model %s (%s) failed to load: %s", key, name, exc)
                continue
            after = _rss_mb()
            _models[key] = model
            _info[key] = {
                "model": name,
                "fallback_used": i > 0,
                "load_seconds": round(time.time() - t0, 1),
                "rss_delta_mb": round(after - before) if before and after else None,
            }
            logger.info("Loaded %s: %s", key, _info[key])
            return model
        _info[key] = {"model": None, "error": str(last_exc)}
        raise RuntimeError(f"No model could be loaded for {key}: {last_exc}") from last_exc


def get_loaded(key: str):
    return _models.get(key)


def loaded_models_info() -> dict:
    return dict(_info)
