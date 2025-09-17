# repo-fastapi — Setup & Development Guide

> This README covers environment setup, running locally, code quality (lint/format/pre-commit), and generating architecture diagrams (Mermaid) on Windows/WSL/macOS/Linux.

## Table of Contents
- [Requirements](#requirements)
- [Quick Setup](#quick-setup)
- [Run Locally & Tests](#run-locally--tests)
- [Generate the Architecture Diagram](#generate-the-architecture-diagram)
- [Code Quality: Ruff / Flake8 / Pre-commit](#code-quality-ruff--flake8--pre-commit)
- [Troubleshooting](#troubleshooting)
- [Windows Tips](#windows-tips)

---

## Requirements
- **Python 3.11+** (64‑bit recommended).
- **Node.js + npm** *(optional)* — only if you want PNG export via Mermaid CLI (`mmdc`).
- Optional developer tools: `git`, PowerShell or Bash, VS Code.

> **Note:** Always use `python` from the project virtualenv `.venv` (avoid `py` on Windows to prevent bypassing the venv).

---

## Quick Setup

### 1) Create & activate the virtual environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
```
**macOS/Linux/WSL:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

### 2) Install application requirements
```powershell
python -m pip install -r requirements.txt
```
If you see import/version conflicts (notably **Pydantic v2** or **Uvicorn/Click**), install:
```powershell
python -m pip install -U "typing_extensions>=4.12.2" "click>=8.1.7"
```

### 3) Development requirements
```powershell
python -m pip install -r requirements-dev.txt
```

---

## Run Locally & Tests
- Smoke test:
```powershell
python -m tests.test_run
```
- Example dev server:
```powershell
uvicorn app.main:app --reload
```

---

## Generate the Architecture Diagram

We ship a comprehensive generator script: `gen_arch.py`.

### Basic usage
```powershell
python gen_arch.py --colored --direction TB --html
```
This produces the following under `docs/`:
- `ARCHITECTURE.md` — main report (Buckets + Runtime/Infra + optional Import Graph + optional Endpoints).
- `architecture.mmd` — raw Mermaid (no code fences).
- `architecture.html` — browser-friendly view (does **not** require `mmdc`).
- `architecture.png` — *(if Mermaid CLI is installed)*.
- `runtime.mmd` — runtime/infra diagram.
- `imports.mmd` — *(optional)* when using `--imports`.
- `endpoints.md` — *(optional)* when using `--endpoints`.

### Handy options
- `--colored` : distinct colors per bucket.
- `--direction LR|TB|BT|RL` : diagram flow (default LR).
- `--imports --imports-max 50` : Python import graph (top-level collapse).
- `--endpoints` : extract FastAPI/Flask routes.
- `--scale 2 --bg white --theme neutral` : PNG quality & themeing.

### PNG export (Mermaid CLI)
Install Mermaid CLI globally:
```powershell
npm install -g @mermaid-js/mermaid-cli
```
Then, for manual export:
```powershell
mmdc -i docs\architecture.mmd -o docs\architecture.png -s 2 --backgroundColor transparent
```
> If `mmdc` is not available, use `--html` and open `docs/architecture.html`.

---

## Code Quality: Ruff / Flake8 / Pre-commit

### Ruff
```powershell
ruff check --fix .
ruff format .
```

### Flake8 (optional)
If CI runs Flake8, you can mirror it locally:
```powershell
flake8 .
```

### Pre-commit
Install and run:
```powershell
pre-commit install
pre-commit run -a
pre-commit autoupdate
```
If pre-commit modifies files, stage them again with `git add`, then commit.

#### Suggested hooks
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.13.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: mixed-line-ending
        args: [--fix=lf]
      - id: check-yaml
```

---

## Troubleshooting

### 1) `AttributeError: module 'click' has no attribute 'Choice'`
Cause: old `click` or a local `click.py` shadowing the package.
Fix:
```powershell
python -m pip install -U "click>=8.1.7" "uvicorn>=0.30"
Get-ChildItem -Recurse -File -Filter click.py   # ensure no local shadowing
```

### 2) `ImportError: cannot import name 'deprecated' from 'typing_extensions'`
Occurred with **Pydantic v2** when `typing_extensions` is too old.
Fix:
```powershell
python -m pip install -U "typing_extensions>=4.12.2" "pydantic>=2.8,<3"
```

### 3) Pre-commit issues (identify / cfgv / virtualenv / filelock)
- Ensure no local shadowing inside the repo:
```powershell
Get-ChildItem -Recurse -File -Filter identify.py
Get-ChildItem -Recurse -Directory -Filter identify
Get-ChildItem -Recurse -File -Filter cfgv.py
Get-ChildItem -Recurse -Directory -Filter cfgv
Get-ChildItem -Recurse -File -Filter filelock.py
Get-ChildItem -Recurse -Directory -Filter filelock
```
- Reinstall compatible versions and clear cache:
```powershell
python -m pip install -U "pre-commit>=3.7" "cfgv>=3.4.0" "identify>=2.5" "virtualenv>=20.26.6" "filelock>=3.12.4"
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\pre-commit" -ErrorAction SilentlyContinue
pre-commit autoupdate
pre-commit run -a
```

### 4) Common Ruff findings
- **E501 (line too long):** split long lines.
- **E701/E702:** avoid multiple statements/returns on a single line.
- **W291:** trailing whitespace — enable `trailing-whitespace` hook.

---

## Windows Tips
- Prefer `python` over `py` to ensure the venv is used.
- Quick open artifacts:
```powershell
Start-Process docs\architecture.png
Start-Process docs\architecture.html
```
- Install Node (if needed), e.g. via winget:
```powershell
winget install OpenJS.NodeJS.LTS
```

---

> **Contributing:** PRs welcome — especially improvements to `gen_arch.py` or the dev tooling.
