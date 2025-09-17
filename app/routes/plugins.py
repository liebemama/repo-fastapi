from __future__ import annotations

import importlib
import inspect
from collections.abc import Iterable
from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


router = APIRouter(prefix="/plugins", tags=["plugins"])


class PluginMeta(BaseModel):
    name: str
    provider: str | None = None
    tasks: list[str] = Field(default_factory=list)


# ----------------------------
# Loader helpers (lazy + initialize if available)
# ----------------------------
def _loader_module():
    mod = importlib.import_module("app.plugins.loader")
    # حاول استدعاء أي دالة تهيئة إن وُجدت
    for fn_name in (
        "ensure_plugins_loaded",
        "load_all_plugins",
        "load_plugins",
        "init_plugins",
        "initialize",
        "discover_plugins",
    ):
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            try:
                fn()
                break
            except Exception:
                pass
    return mod


def _iter_plugin_instances() -> Iterable[Any]:
    loader = _loader_module()

    for name in (
        "get_available_plugins",
        "list_available_plugins",
        "available_plugins",
        "get_plugins",
        "iter_plugins",
    ):
        fn = getattr(loader, name, None)
        if callable(fn):
            try:
                plugins = fn()
                if plugins:
                    return plugins
            except Exception:
                pass

    for reg_name in ("REGISTRY", "PLUGINS", "plugins", "registry"):
        reg = getattr(loader, reg_name, None)
        if isinstance(reg, dict) and reg:
            return reg.values()
        if isinstance(reg, (list | tuple)) and reg:
            return reg

    for name_api in ("get_plugin_names", "list_plugin_names", "available_plugin_names"):
        fn = getattr(loader, name_api, None)
        if callable(fn):
            try:
                names = fn()
                if names:
                    get_inst = getattr(loader, "get_plugin_instance", None)
                    if callable(get_inst):
                        return filter(None, (get_inst(n) for n in names))
            except Exception:
                pass

    return ()


def _instantiate_direct(name: str) -> Any | None:
    """
    Fallback صارم: استورد app.plugins.<name>.plugin وابنِ Plugin().
    """
    try:
        mod = importlib.import_module(f"app.plugins.{name}.plugin")
    except Exception:
        return None

    plugin_cls = getattr(mod, "Plugin", None)
    if plugin_cls is None:
        return None

    try:
        inst = plugin_cls()  # type: ignore[call-arg]
    except Exception:
        return None

    # نفّذ load() إن وُجدت
    try:
        load_fn = getattr(inst, "load", None)
        if callable(load_fn):
            load_fn()
    except Exception:
        # حتى لو فشلت الـ load، نرجّع الـ instance (قد تعمل المهام الخفيفة)
        pass

    # اسم افتراضي لو غير مضمّن
    if not getattr(inst, "name", None):
        try:
            inst.name = name
        except Exception:
            pass

    return inst


def _get_plugin_instance(name: str) -> Any | None:
    loader = _loader_module()

    for fn_name in ("get_plugin_instance", "load_plugin", "get", "resolve_plugin"):
        fn = getattr(loader, fn_name, None)
        if callable(fn):
            try:
                inst = fn(name)
                if inst is not None:
                    return inst
            except Exception:
                pass

    for reg_name in ("REGISTRY", "PLUGINS", "plugins", "registry"):
        reg = getattr(loader, reg_name, None)
        if isinstance(reg, dict) and name in reg:
            return reg[name]

    # Fallback مباشر على باكيج البلجن
    return _instantiate_direct(name)


def _serialize_meta(plugin: Any) -> PluginMeta:
    name = getattr(plugin, "name", None) or getattr(getattr(plugin, "__class__", None), "__name__", "unknown")
    provider = getattr(plugin, "provider", None)
    tasks_attr = getattr(plugin, "tasks", None)
    tasks = list(tasks_attr) if isinstance(tasks_attr, (list | tuple | set)) else []
    return PluginMeta(name=str(name), provider=provider, tasks=tasks)


# ----------------------------
# Routes
# ----------------------------
@router.get(
    "",
    response_model=list[PluginMeta],
    summary="List all plugins",
    description="Returns a list of available plugins with basic metadata.",
)
def list_plugins() -> list[PluginMeta]:
    instances = list(_iter_plugin_instances())
    return [_serialize_meta(p) for p in instances]


@router.get(
    "/{name}",
    response_model=PluginMeta,
    summary="Get plugin metadata",
    description="Returns metadata for a specific plugin.",
)
def get_plugin(
    name: Annotated[str, Path(min_length=1, description="Plugin name (folder name).")],
) -> PluginMeta:
    plugin = _get_plugin_instance(name)
    if plugin is None:
        raise HTTPException(status_code=404, detail=f"Plugin not found: {name}")
    return _serialize_meta(plugin)


@router.post(
    "/{name}/{task}",
    summary="Run a task of a plugin",
    description="Executes a task for a given plugin with an arbitrary JSON payload.",
)
async def run_plugin_task(
    name: Annotated[str, Path(min_length=1, description="Plugin name (folder name).")],
    task: Annotated[str, Path(min_length=1, description="Task name exposed by the plugin.")],
    payload: Annotated[dict[str, Any], Body(..., description="Arbitrary JSON payload for the task.")],
) -> JSONResponse:
    """
    Resolution order:
      1) Call `plugin.<task>(payload)` if callable.
      2) Otherwise, if `plugin.infer` exists, call it with the task embedded into payload.
    """
    plugin = _get_plugin_instance(name)
    if plugin is None:
        raise HTTPException(status_code=404, detail=f"Plugin not found: {name}")

    declared_tasks = getattr(plugin, "tasks", [])
    fn = getattr(plugin, task, None)

    # المسار (1): ميثود بنفس اسم المهمة إن وُجدت
    if callable(fn):
        try:
            if inspect.iscoroutinefunction(fn):
                result = await fn(payload)  # type: ignore[misc]
            else:
                result = fn(payload)  # type: ignore[misc]
            return JSONResponse({"plugin": name, "task": task, "result": result})
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Task '{task}' failed: {e!s}") from e

    # المسار (2): نقطة دخول عامة infer
    infer_fn = getattr(plugin, "infer", None)
    if callable(infer_fn):
        # مرّر اسم المهمة داخل الـ payload حتى يقدر البلَجن يفرّق بين الـ subtasks
        forwarded = dict(payload)
        forwarded.setdefault("task", task)
        try:
            if inspect.iscoroutinefunction(infer_fn):
                result = await infer_fn(forwarded)  # type: ignore[misc]
            else:
                result = infer_fn(forwarded)  # type: ignore[misc]
            return JSONResponse({"plugin": name, "task": task, "result": result})
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Infer for '{task}' failed: {e!s}") from e

    # لو لا دي ولا دي، يبقى المهمة غير متاحة فعلًا
    available = list(declared_tasks) if isinstance(declared_tasks, (list | tuple | set)) else []
    raise HTTPException(
        status_code=404,
        detail=f"Task '{task}' not found in plugin '{name}'. Available: {available or ['<none>']}",
    )
