# 🧹 Code Style & Linting Guide

This project enforces a **clean, consistent, and modern code style** using **[Ruff](https://docs.astral.sh/ruff/)**.
Ruff is an all-in-one tool that replaces **flake8**, **isort**, and **black**, making it fast and simple to use.

---

## ✨ Why Ruff?
- 🚀 **Fast** — written in Rust, lightning quick.
- 🧩 **All-in-one** — linter, import sorter, and formatter.
- 🎯 **PEP8-compliant** with additional useful rules.
- 🛠️ **Auto-fix** common issues instantly.

---

## ⚙️ Configuration

All settings are stored in the [`pyproject.toml`](../pyproject.toml) file:

- Target version: **Python 3.12**
- Line length: **120**
- Enabled rules: `E`, `F`, `I`, `UP`, `B`
- Import sorting (`isort`-style) built into Ruff
- Black-compatible formatting

Per-file ignores are defined for long lines in `tests/` and `scripts/`.

---

## 🔧 Commands

### Check for issues
```bash
ruff check .
```

### Auto-fix issues
```bash
ruff check . --fix
```

### Format code (like Black)
```bash
ruff format .
```

---

## 🖥️ Helper Scripts

Instead of typing commands manually, you can use the provided scripts:

- **Windows (PowerShell):**
  ```powershell
  pwsh run_lint.ps1
  ```
  Or check only (no changes):
  ```powershell
  pwsh run_lint.ps1 -CheckOnly
  ```

- **Linux/macOS (Bash):**
  ```bash
  bash run_lint.sh
  ```
  Or check only:
  ```bash
  bash run_lint.sh --check
  ```

---

## 🔗 Pre-commit Hook (Optional)

To enforce linting before every commit, use [pre-commit](https://pre-commit.com/):

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

The configuration is defined in [`.pre-commit-config.yaml`](../.pre-commit-config.yaml).

---

## 🤖 CI Integration

Continuous Integration is set up in [`.github/workflows/ci-ruff.yml`](../.github/workflows/ci-ruff.yml).
It runs `ruff check .` and `ruff format --check .` on every push and pull request to the `main` branch.

---

## ✅ Summary
- **Ruff** is the single source of truth for linting, formatting, and import sorting.
- Run `ruff check . --fix` and `ruff format .` before committing.
- Use helper scripts or pre-commit hooks for automation.
- CI ensures consistent code style across all environments.
