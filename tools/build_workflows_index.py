#!/usr/bin/env python3

"""
Generate docs/workflows-overview.md from workflows metadata
and ensure per-workflow README.md files.

- Scans app/workflows/* for manifest.json + workflow.json
- Auto-creates README.md for any workflow (or refreshes all with --force-readme)
- Idempotent; safe to run anytime.

Usage:
    python tools/build_workflows_index.py
    python tools/build_workflows_index.py --force-readme
"""

from __future__ import annotations

import argparse
import json
import locale
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Paths
ROOT = Path(__file__).resolve().parents[1]  # project root
WF_DIR = ROOT / "app" / "workflows"
OUT_DIR = ROOT / "docs"
OUT_MD = OUT_DIR / "workflows-overview.md"

README_TEMPLATE = """# {name}

**Version:** {version}
**Tags:** {tags_fmt}
**Description:** {description}

## Overview
This workflow is defined by:
- **manifest:** `app/workflows/{folder}/manifest.json`
- **sequence:** `app/workflows/{folder}/{sequence_file}`

**Steps:** {steps_count}

## How it runs

### Option A — by preset name (if you expose it)
POST `/workflow/run`
```json
{{
  "preset": "{name}",
  "inputs": {{}}
}}
```

### Option B — by explicit sequence
POST `/workflow/run`
```json
{sequence_pretty}
```

> The API base for workflows is `/workflow` (e.g. `POST /workflow/run`).

## Notes
- Placeholders like `{{{{ asr.text }}}}` or `{{{{audio_url}}}}` are supported by the router.
- Make sure any referenced plugins are installed/available.
"""

OK = "✅"
DOC = "📄"
UPD = "📝"


def _supports_utf8() -> bool:
    enc = (getattr(sys.stdout, "encoding", None) or "") or locale.getpreferredencoding(False) or ""
    enc_up = enc.upper()
    return "UTF-8" in enc_up or "UTF8" in enc_up


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # py>=3.7
except Exception:
    pass

if not _supports_utf8():
    OK, DOC, UPD = "[OK]", "[DOC]", "[UPD]"


@dataclass
class WorkflowMeta:
    folder: str
    name: str
    version: str
    description: str
    tags: list[str]
    sequence_file: str
    steps_count: int
    manifest_rel: str | None
    sequence_rel: str | None
    sequence_raw: dict[str, Any] | list[Any] | None


def write_if_changed(path: Path, content: str, encoding: str = "utf-8") -> bool:
    old = None
    if path.exists():
        try:
            old = path.read_text(encoding=encoding)
        except Exception:
            old = None
    if old == content:
        return False
    path.write_text(content, encoding=encoding)
    return True


def _pretty_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _extract_steps_count(seq: Any) -> int:
    # We expect {"sequence": [...]} like in asr_clean_ar/workflow.json,
    # but try to be flexible.
    if isinstance(seq, dict):
        if isinstance(seq.get("sequence"), list):
            return len(seq["sequence"])
        if isinstance(seq.get("steps"), list):
            return len(seq["steps"])
        nodes = seq.get("nodes")
        if isinstance(nodes, list):
            return len(nodes)
        if isinstance(nodes, dict):
            return len(nodes)
        return 0
    if isinstance(seq, list):
        return len(seq)
    return 0


def collect_workflows() -> list[WorkflowMeta]:
    out: list[WorkflowMeta] = []
    if not WF_DIR.exists():
        return out

    candidates = [p for p in WF_DIR.iterdir() if p.is_dir() and not p.name.startswith(".") and p.name != "__pycache__"]

    for wdir in sorted(candidates, key=lambda p: p.name.lower()):
        manifest = wdir / "manifest.json"

        if not manifest.exists():
            # consider it a workflow only if manifest present
            continue

        try:
            m = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            m = {}

        name = str(m.get("name") or wdir.name)
        version = str(m.get("version") or "0.1.0")
        description = str(m.get("description") or "—")
        tags = list(m.get("tags") or [])
        sequence_file = str(m.get("sequence_file") or "workflow.json")

        sequence_path = wdir / sequence_file
        seq_raw: Any = None
        if sequence_path.exists():
            try:
                seq_raw = json.loads(sequence_path.read_text(encoding="utf-8"))
            except Exception:
                seq_raw = None

        steps_count = _extract_steps_count(seq_raw)

        out.append(
            WorkflowMeta(
                folder=wdir.name,
                name=name,
                version=version,
                description=description,
                tags=tags,
                sequence_file=sequence_file,
                steps_count=steps_count,
                manifest_rel=manifest.relative_to(ROOT).as_posix(),
                sequence_rel=sequence_path.relative_to(ROOT).as_posix() if sequence_path.exists() else None,
                sequence_raw=seq_raw,
            )
        )
    return out


def render_readme(meta: WorkflowMeta) -> str:
    tags_fmt = ", ".join(meta.tags) if meta.tags else "—"
    # Construct a minimal explicit payload for Option B
    # If stored form has {"sequence": [...]}, reuse it; else wrap.
    seq_payload: dict[str, Any]
    if isinstance(meta.sequence_raw, dict) and "sequence" in meta.sequence_raw:
        seq_payload = meta.sequence_raw
    else:
        seq_payload = {"sequence": meta.sequence_raw or []}

    return (
        README_TEMPLATE.format(
            name=meta.name,
            version=meta.version,
            tags_fmt=tags_fmt,
            description=meta.description or "—",
            folder=meta.folder,
            sequence_file=meta.sequence_file,
            steps_count=meta.steps_count,
            sequence_pretty=_pretty_json(seq_payload),
        ).rstrip()
        + "\n"
    )


def ensure_readme(meta: WorkflowMeta, force: bool = False) -> bool:
    """Create/refresh app/workflows/<name>/README.md"""
    readme_path = ROOT / "app" / "workflows" / meta.folder / "README.md"
    if readme_path.exists() and not force:
        return False
    readme_path.write_text(render_readme(meta), encoding="utf-8")
    return True


def render_index_md(items: list[WorkflowMeta]) -> str:
    lines: list[str] = [
        "# 🧭 Workflows Index",
        "",
        "_Auto-generated. Do not edit manually._",
        "",
        "| Workflow | Version | Steps | Tags | README | manifest.json | workflow.json |",
        "|---------:|:--------|:-----:|:-----|:------:|:-------------:|:-------------:|",
    ]
    for meta in items:
        tags = ", ".join(meta.tags) if meta.tags else "—"
        readme_rel = f"app/workflows/{meta.folder}/README.md"
        readme_link = f"[README](../{readme_rel})"
        manifest_link = f"[manifest](../{meta.manifest_rel})" if meta.manifest_rel else "—"
        seq_link = f"[sequence](../{meta.sequence_rel})" if meta.sequence_rel else "—"

        lines.append(
            f"| **{meta.name}** | {meta.version} | {meta.steps_count} | {tags} | "
            f"{readme_link} | {manifest_link} | {seq_link} |"
        )

    lines += [
        "",
        "## How to generate",
        "```bash",
        "python tools/build_workflows_index.py",
        "```",
        "",
        "## Force-refresh all workflow READMEs",
        "```bash",
        "python tools/build_workflows_index.py --force-readme",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build workflows index and auto-generate per-workflow README files.")
    parser.add_argument("--force-readme", action="store_true", help="Regenerate README.md for all workflows.")
    args = parser.parse_args(argv)

    workflows = collect_workflows()

    created = 0
    updated = 0
    for meta in workflows:
        changed = ensure_readme(meta, force=args.force_readme)
        if changed:
            if args.force_readme:
                updated += 1
            else:
                created += 1

    # re-collect to capture new README paths if needed
    workflows = collect_workflows()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_index_md(workflows), encoding="utf-8")

    print(f"{OK} Wrote {OUT_MD.relative_to(ROOT)} ({len(workflows)} workflows)")
    if created:
        print(f"{DOC} Created {created} README file(s)")
    if updated:
        print(f"{UPD} Updated {updated} README file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
