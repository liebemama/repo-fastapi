from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.plugins.loader import get_plugin_instance
from app.workflows import registry as wf


# استيراد اختياري لـ list_plugins (قد لا تكون متاحة في بعض الإصدارات)
try:
    from app.plugins.loader import list_plugins as _list_plugins  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _list_plugins = None  # fallback

router = APIRouter(tags=["workflow"])


# =========================
# Models
# =========================


class Step(BaseModel):
    """A single step in the workflow."""

    name: str = Field(..., min_length=1)
    plugin: str = Field(..., min_length=1)
    task: str = Field(..., min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    timeout: int | None = None  # optional per-step timeout (not enforced here)


class WorkflowRequest(BaseModel):
    """Request body for running a workflow."""

    # Option A: explicit sequence
    sequence: list[Step] | None = None
    # Option B: use a preset by name
    preset: str | None = None
    # Option C: automatic sequence (simple heuristic)
    auto: bool = False

    # Optional high-level inputs, used by auto/presets and placeholder injection
    inputs: dict[str, Any] | None = None
    audio_url: str | None = None
    language: str | None = None
    max_new_tokens: int | None = 256

    # 'return' is reserved in Python; expose via alias
    return_: str | None = Field(default=None, alias="return")


# =========================
# Presets (examples)
# =========================

# Placeholders like "{audio_url}" or "{asr.text}" will be injected later.
PRESETS: dict[str, list[Step]] = {
    "arabic_asr_plus": [
        Step(
            name="asr",
            plugin="whisper",
            task="speech-to-text",
            payload={
                "audio_url": "{audio_url}",
                "language": None,  # auto-detect
                "max_new_tokens": 256,
                # if the plugin supports internal postprocess
                "postprocess": True,
            },
            timeout=180,
        ),
        # Examples if you add text plugins:
        # Step(
        #     name="normalize",
        #     plugin="text_tools",
        #     task="arabic_normalize",
        #     payload={"text": "{asr.text}"}
        # ),
        # Step(
        #     name="spellcheck",
        #     plugin="text_tools",
        #     task="spellcheck_ar",
        #     payload={"text": "{normalize.text}"}
        # ),
    ],
}


# =========================
# Helpers
# =========================


def _lookup_path(obj: Any, dotted: str) -> Any:
    """
    Resolve a dotted path like "asr.text" inside a nested dict.
    Returns None if not found.
    """
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _inject_placeholders(value: Any, context: dict[str, Any]) -> Any:
    """
    Recursively replace string placeholders like "{asr.text}" with values from the context.
    Supports dicts, lists, and strings. Other types are returned as-is.

    Note: This replaces only whole-string placeholders. To support multiple
    placeholders inside one string, extend with a regex-based replacer.
    """
    if isinstance(value, dict):
        return {k: _inject_placeholders(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_inject_placeholders(v, context) for v in value]
    if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
        key = value[1:-1]
        # special-case: top-level inputs or direct fields in context_root
        if key in context.get("_root_", {}):
            return context["_root_"][key]
        # dotted reference into named step outputs (e.g., asr.text)
        if "." in key:
            step_name, path = key.split(".", 1)
            step_obj = context.get(step_name)
            if isinstance(step_obj, dict):
                return _lookup_path(step_obj, path)
        # plain step name -> raw step dict
        return context.get(key)
    return value


def _build_auto_sequence(req: WorkflowRequest) -> list[Step]:
    """
    Very simple heuristic: if audio_url exists, run Whisper.
    Extend here to detect other input types automatically.
    """
    if req.audio_url:
        return [
            Step(
                name="asr",
                plugin="whisper",
                task="speech-to-text",
                payload={
                    "audio_url": "{audio_url}",
                    "language": req.language,
                    "max_new_tokens": req.max_new_tokens or 256,
                    "postprocess": True,
                },
                timeout=180,
            )
        ]
    raise HTTPException(status_code=400, detail="auto mode requires 'audio_url' (or provide a preset/sequence)")


def _resolve_sequence(req: WorkflowRequest) -> tuple[list[Step], str | None]:
    """
    Resolve which sequence to run and return (steps, return_step_name).
    Priority:
      1) explicit sequence
      2) preset from filesystem (workflow.json)
      3) preset from in-code PRESETS
      4) auto (if possible)
    """
    # 1) explicit sequence
    if req.sequence:
        return (list(req.sequence), None)

    # 2) preset: try from filesystem first
    if req.preset:
        try:
            wf_def = wf.get_workflow(req.preset)  # dict loaded from workflow.json
            steps = [Step(**s) for s in wf_def.get("sequence", [])]
            ret = wf_def.get("return")
            return (steps, ret)
        except Exception:
            pass  # fallback to in-code presets below

        steps = PRESETS.get(req.preset)
        if not steps:
            raise HTTPException(404, f"Preset '{req.preset}' not found")
        return ([Step(**s.model_dump()) for s in steps], None)

    # 3) auto
    if req.auto or req.audio_url:
        return (_build_auto_sequence(req), None)

    # 4) invalid
    raise HTTPException(400, "Provide one of: 'sequence', 'preset', or 'auto' (with suitable inputs).")


def _get_available_plugins() -> set[str]:
    """Return available plugin names as a set, regardless of the registry format."""
    if _list_plugins is None:
        return set()
    try:
        reg = _list_plugins()
    except Exception:
        return set()
    if isinstance(reg, dict):
        return set(reg.keys())
    if isinstance(reg, list | tuple | set):  # Ruff UP038-compliant
        return set(reg)
    return set()


def _validate_sequence(seq: list[Step]) -> None:
    """Validate plugins exist before execution; raise 400 with helpful info."""
    available = _get_available_plugins()
    if not available:
        return
    for i, st in enumerate(seq, 1):
        if not st.plugin or st.plugin not in available:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Plugin '{st.plugin}' not found at step #{i}",
                    "available_plugins": sorted(available),
                },
            )


def _run_step(step: Step, context: dict[str, Any]) -> dict[str, Any]:
    """Resolve placeholders and invoke the selected plugin."""
    try:
        plugin = get_plugin_instance(step.plugin)
    except KeyError as e:
        # Prefer 404; frontends can distinguish 'not found' from 'bad request'
        raise HTTPException(status_code=404, detail=f"Plugin '{step.plugin}' not found") from e

    tasks = getattr(plugin, "tasks", None)
    if callable(tasks):
        tasks = tasks()
    if isinstance(tasks, list | tuple | set) and step.task not in tasks:  # Ruff UP038-compliant
        raise HTTPException(400, f"Plugin '{step.plugin}' does not support task '{step.task}'")

    # Merge a small root context for top-level fields (e.g., audio_url)
    root_inputs = dict(context.get("_root_", {}))
    payload = _inject_placeholders(step.payload, {**context, "_root_": root_inputs})

    try:
        result = plugin.infer(payload)  # type: ignore[attr-defined]
    except Exception as err:
        # Keep original traceback (ruff B904-friendly)
        raise HTTPException(status_code=500, detail=f"Step '{step.name}' failed: {err!s}") from err

    return result if isinstance(result, dict) else {"result": result}


# =========================
# Routes
# =========================


@router.get("/ping")
def workflow_ping() -> dict[str, Any]:
    return {"ok": True}


@router.get("/presets")
def list_presets():
    file_presets: list[str] = []
    try:
        file_presets = [w["name"] for w in wf.list_workflows()]
    except Exception:
        pass
    code_presets = list(PRESETS.keys())
    return {"ok": True, "presets": sorted(set(file_presets + code_presets))}


@router.post("/run")
def run_workflow(req: WorkflowRequest) -> dict[str, Any]:
    """
    Run a workflow by:
      - explicit `sequence`, or
      - `preset` (filesystem or in-code), or
      - `auto` (simple heuristic; currently Whisper if `audio_url` provided).
    Placeholders like "{audio_url}" or "{asr.text}" are supported.
    """
    sequence, preset_return = _resolve_sequence(req)

    # Validate before execution (clear 400 instead of 500 later)
    _validate_sequence(sequence)

    target_return = req.return_ or preset_return

    # Root-level context (available for placeholders via {_root_: ...})
    root_context: dict[str, Any] = {}
    if req.inputs:
        root_context.update(req.inputs)
    if req.audio_url is not None:
        root_context["audio_url"] = req.audio_url
    if req.language is not None:
        root_context["language"] = req.language
    if req.max_new_tokens is not None:
        root_context["max_new_tokens"] = req.max_new_tokens

    context: dict[str, Any] = {"_root_": root_context}
    results: list[dict[str, Any]] = []

    if not sequence:
        raise HTTPException(400, "Empty workflow sequence.")

    for step in sequence:
        out = _run_step(step, context)
        # Save into context under step.name for downstream references
        context[step.name] = out
        results.append({"step": step.name, "plugin": step.plugin, "task": step.task, "output": out})

    if target_return:
        for r in results:
            if r["step"] == target_return:
                return {"ok": True, "result": r["output"]}
        # return target specified but not found in steps
        raise HTTPException(400, f"return target '{target_return}' not found in steps")

    return {"ok": True, "count": len(results), "results": results}
