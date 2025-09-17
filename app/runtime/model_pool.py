# app/runtime/model_pool.py
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import torch

from app.core.config import get_settings


class ModelPool:
    """
    Keeps up to `max_active` models loaded (e.g., on GPU).
    Evicts least-recently-used models and unloads idle ones.
    Thread-safe.
    """

    def __init__(self, max_active: int = 2, idle_unload_s: int = 600):
        self.max_active = max_active
        self.idle_unload_s = idle_unload_s
        self.pool: OrderedDict[str, dict] = OrderedDict()  # name -> {"model": obj, "last": ts}
        self.lock = threading.Lock()

    def get(self, name: str, factory: Callable[[], Any]):
        """
        Get (or lazily create) a model by name.
        `factory` must return a ready-to-use model (moved to device, eval(), dtype set, etc.).
        """
        now = time.time()
        with self.lock:
            if name in self.pool:
                self.pool[name]["last"] = now
                self.pool.move_to_end(name)
                return self.pool[name]["model"]

            # Create/load new model
            model = factory()
            self.pool[name] = {"model": model, "last": now}
            self.pool.move_to_end(name)

            # Evict LRU while exceeding max_active
            while self.max_active > 0 and len(self.pool) > self.max_active:
                old_key, item = self.pool.popitem(last=False)
                self._safe_del(item.get("model"))

            self._empty_cuda_cache()
            return model

    def sweep_idle(self):
        """
        Unload models that have been idle longer than `idle_unload_s`.
        """
        if self.idle_unload_s <= 0:
            return
        now = time.time()
        with self.lock:
            to_drop = [k for k, v in self.pool.items() if now - v["last"] > self.idle_unload_s]
            for k in to_drop:
                item = self.pool.pop(k)
                self._safe_del(item.get("model"))
        self._empty_cuda_cache()

    @staticmethod
    def _safe_del(model):
        try:
            del model
        except Exception:
            pass

    @staticmethod
    def _empty_cuda_cache():
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


@lru_cache
def get_model_pool() -> ModelPool:
    """
    Global singleton pool configured from Settings.
    Safe to import from plugins without circular imports.
    """
    s = get_settings()
    return ModelPool(
        max_active=s.MAX_ACTIVE_MODELS,
        idle_unload_s=s.IDLE_UNLOAD_SECONDS,
    )
