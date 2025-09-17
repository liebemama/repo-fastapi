from __future__ import annotations

import importlib.util
import json
import logging
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.plugins.base import AIPlugin


logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Internal state
# --------------------------------------------------------------------
_PLUGINS_DIR = Path(__file__).parent
_REGISTRY: dict[str, AIPlugin] = {}  # name -> Plugin instance (loaded)
_META: dict[str, dict[str, Any]] = {}  # name -> metadata (from manifest / fs)
_LOADED: bool = False
_LOCK = threading.RLock()


# --------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------
@dataclass(frozen=True)
class PluginPath:
    name: str
    folder: Path
    plugin_file: Path
    manifest_file: Path


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _is_viable_plugin_dir(d: Path) -> bool:
    """A viable plugin dir is a visible directory that contains plugin.py."""
    if not d.is_dir():
        return False
    if d.name in {"__pycache__"} or d.name.startswith(".") or d.name.startswith("_"):
        return False
    return (d / "plugin.py").is_file()


def _scan_fs() -> list[PluginPath]:
    """Scan the filesystem for plugin folders deterministically (sorted)."""
    if not _PLUGINS_DIR.exists():
        return []
    items: list[PluginPath] = []
    for d in sorted(_PLUGINS_DIR.iterdir(), key=lambda p: p.name.lower()):
        if _is_viable_plugin_dir(d):
            items.append(
                PluginPath(
                    name=d.name,
                    folder=d,
                    plugin_file=d / "plugin.py",
                    manifest_file=d / "manifest.json",
                )
            )
    return items


def _read_manifest(p: PluginPath) -> dict[str, Any] | None:
    """Read manifest.json if present; return None if missing or invalid."""
    if not p.manifest_file.exists():
        return None
    try:
        return json.loads(p.manifest_file.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to parse manifest for %s: %s", p.name, e)
        return None


def _build_meta_entry(p: PluginPath, manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Compose a metadata dict for the plugin without importing it."""
    # Name priority: manifest.name -> folder name
    name = (manifest or {}).get("name") or p.name

    meta: dict[str, Any] = {
        "name": name,
        "folder": str(p.folder),
        "plugin_file": str(p.plugin_file),
        "has_manifest": manifest is not None,
    }
    if manifest:
        # Copy a subset of common manifest fields if present
        for k in ("version", "provider", "description", "tasks", "homepage"):
            if k in manifest:
                meta[k] = manifest[k]
        meta["manifest"] = manifest
    return meta


def _module_name_for(name: str) -> str:
    """Stable module name under the app.plugins.* namespace."""
    return f"app.plugins.{name}"


def _import_plugin_module(name: str, plugin_file: Path):
    """Import a plugin module from its plugin.py path under a stable module name."""
    module_name = _module_name_for(name)
    spec = importlib.util.spec_from_file_location(module_name, str(plugin_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for plugin '{name}' from {plugin_file}")
    module = importlib.util.module_from_spec(spec)
    # Pre-register to support relative imports inside the plugin, if any
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------
def discover(reload: bool = False) -> dict[str, dict[str, Any]]:
    """
    Discover plugin metadata from the filesystem (without importing heavy code).
    Cached after the first call unless reload=True.
    Returns a dict: name -> metadata.
    """
    global _LOADED, _META
    with _LOCK:
        if _LOADED and not reload:
            return _META

        # If reloading, clear both meta and any previously loaded instances
        if reload:
            _META = {}
            _REGISTRY.clear()

        for p in _scan_fs():
            manifest = _read_manifest(p)
            meta = _build_meta_entry(p, manifest)
            # Use folder name as the stable key in the registry/meta
            _META[p.name] = meta

        _LOADED = True
        return _META


def ensure_loaded() -> None:
    """Ensure that discovery has been performed at least once."""
    discover(reload=False)


def get(name: str) -> AIPlugin:
    """
    Get (and if needed, load) a plugin instance by its folder name or manifest name.
    - Primary key is the folder name (stable). If caller passes a manifest 'name',
      we map it back to folder name.
    """
    ensure_loaded()
    with _LOCK:
        # Map given name to folder key if user passed manifest.name
        folder_key = name
        if name not in _META:
            # Try reverse lookup by manifest "name"
            for k, m in _META.items():
                if m.get("name") == name:
                    folder_key = k
                    break

        meta = _META.get(folder_key)
        if meta is None:
            raise KeyError(f"Plugin '{name}' not found")

        if folder_key in _REGISTRY:
            return _REGISTRY[folder_key]

        # Import and instantiate
        plugin_file = Path(meta["plugin_file"])
        try:
            module = _import_plugin_module(folder_key, plugin_file)
        except Exception as e:
            # keep traceback visible in logs but raise a concise error
            logger.error("Failed to import plugin '%s':\n%s", folder_key, traceback.format_exc())
            raise ImportError(f"Failed to import plugin '{folder_key}'") from e

        if not hasattr(module, "Plugin"):
            raise ImportError(f"Plugin module '{folder_key}' has no class 'Plugin'")

        plugin_cls = module.Plugin
        if not isinstance(plugin_cls, type) or not issubclass(plugin_cls, AIPlugin):
            raise TypeError(f"'Plugin' in '{folder_key}' must inherit AIPlugin")

        try:
            instance: AIPlugin = plugin_cls()
            # Defensive: some plugins may expect load() to initialize resources.
            instance.load()
        except Exception as e:
            logger.error("Failed to initialize plugin '%s':\n%s", folder_key, traceback.format_exc())
            raise RuntimeError(f"Failed to initialize plugin '{folder_key}'") from e

        _REGISTRY[folder_key] = instance
        return instance


def all_meta() -> list[dict[str, Any]]:
    """Return a sorted list of plugin metadata (by visible name)."""
    ensure_loaded()
    items = list(_META.values())
    items.sort(key=lambda m: str(m.get("name", "")))
    return items


def exists(name: str) -> bool:
    """Check whether a plugin exists (by folder or manifest name)."""
    ensure_loaded()
    if name in _META:
        return True
    return any(m.get("name") == name for m in _META.values())


def unload(name: str) -> bool:
    """Remove a loaded instance from the registry (does not delete files)."""
    with _LOCK:
        return _REGISTRY.pop(name, None) is not None


def reload_all() -> None:
    """Full reload: re-scan metadata and drop all loaded instances."""
    discover(reload=True)


# --------------------------------------------------------------------
# Backward-compat aliases (keep older imports working)
# --------------------------------------------------------------------
def get_available_plugins() -> list[dict[str, Any]]:
    """
    Legacy helper used by older routes/tests.
    Returns a list of plugin metadata dicts (sorted).
    """
    return all_meta()


def get_plugin_instance(name: str) -> AIPlugin:
    """
    Legacy alias to fetch a plugin instance by name.
    """
    return get(name)


__all__ = [
    "discover",
    "ensure_loaded",
    "get",
    "all_meta",
    "exists",
    "unload",
    "reload_all",
    # Compatibility
    "get_available_plugins",
    "get_plugin_instance",
]
