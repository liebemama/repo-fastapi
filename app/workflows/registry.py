from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, field_validator


WORKFLOWS_ROOT = Path(__file__).resolve().parent  # app/workflows


class WorkflowManifest(BaseModel):
    name: str
    version: str = "0.1.0"
    description: str | None = None
    sequence_file: str = "workflow.json"
    tags: list[str] = []

    @field_validator("name")
    @classmethod
    def _no_spaces(cls, v: str) -> str:
        if " " in v:
            raise ValueError("workflow name must not contain spaces")
        return v


class WorkflowSpec(BaseModel):
    manifest: WorkflowManifest
    sequence: dict  # محتوى workflow.json بالكامل


_REGISTRY: dict[str, WorkflowSpec] = {}
_LOADED = False


def load_all(root: Path | None = None) -> None:
    global _LOADED
    if root is None:
        root = WORKFLOWS_ROOT
    _REGISTRY.clear()

    for m in root.glob("*/manifest.json"):
        try:
            manifest = WorkflowManifest.model_validate_json(m.read_text(encoding="utf-8"))
        except Exception:
            manifest = WorkflowManifest.model_validate(json.loads(m.read_text(encoding="utf-8")))
        wf_dir = m.parent
        seq_path = wf_dir / manifest.sequence_file
        sequence = json.loads(seq_path.read_text(encoding="utf-8"))
        _REGISTRY[manifest.name] = WorkflowSpec(manifest=manifest, sequence=sequence)
    _LOADED = True


def ensure_loaded() -> None:
    if not _LOADED:
        load_all()


def list_workflows() -> list[dict]:
    ensure_loaded()
    return [
        {
            "name": spec.manifest.name,
            "version": spec.manifest.version,
            "description": spec.manifest.description,
            "tags": spec.manifest.tags,
        }
        for spec in _REGISTRY.values()
    ]


def get_workflow(name: str) -> dict:
    ensure_loaded()
    try:
        return _REGISTRY[name].sequence
    except KeyError:
        raise KeyError(f"Workflow '{name}' not found") from None
