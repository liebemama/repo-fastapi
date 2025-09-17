# Changelog

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.1.0 — 2025-09-13

### Added
- FastAPI app bootstrap with health & env endpoints. (see `app/main.py`)
- Plugin system: base class, loader, and example plugins (`dummy`, `neu_server`).
- Cross‑platform device runtime (CUDA/CPU/MPS/ROCm) helpers.
- Unified response utilities and error/logging setup.
- Cross‑platform CI (Ubuntu, Windows, macOS) + self‑hosted GPU workflow.
- Developer tooling: Ruff, Pytest, Coverage, Pre‑commit.
- Docs: README (with badges table), API.md, ARCHITECTURE.md.

### Changed
- Organized repository structure (`app/`, `scripts/`, `tests/`, `.github/workflows/`, etc.).

### Notes
- Security/auth is intentionally minimal in this release; API‑Key/JWT planned next.
- Docker images and plugin generator CLI planned for upcoming releases.
