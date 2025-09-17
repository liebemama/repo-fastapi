# 🚀 NeuroServe — GPU-Ready FastAPI AI Server

## 📊 Project Status

| Category      | Badges |
|---------------|--------|
| **Languages** | ![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white) ![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white) ![CSS3](https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white) |
| **Framework** | ![FastAPI](https://img.shields.io/badge/FastAPI-0.116.x-009688?logo=fastapi&logoColor=white) |
| **ML / GPU**  | ![PyTorch](https://img.shields.io/badge/PyTorch-2.6.x-EE4C2C?logo=pytorch&logoColor=white) ![CUDA Ready](https://img.shields.io/badge/CUDA-Ready-76B900?logo=nvidia&logoColor=white) |
| **CI**        | [![Ubuntu CI](https://github.com/TamerOnLine/repo-fastapi/actions/workflows/ci-ubuntu.yml/badge.svg)](https://github.com/TamerOnLine/repo-fastapi/actions/workflows/ci-ubuntu.yml) [![Windows CI](https://github.com/TamerOnLine/repo-fastapi/actions/workflows/ci-windows.yml/badge.svg)](https://github.com/TamerOnLine/repo-fastapi/actions/workflows/ci-windows.yml) [![Windows GPU CI](https://github.com/TamerOnLine/repo-fastapi/actions/workflows/ci-gpu.yml/badge.svg)](https://github.com/TamerOnLine/repo-fastapi/actions/workflows/ci-gpu.yml) [![macOS CI](https://github.com/TamerOnLine/repo-fastapi/actions/workflows/ci-macos.yml/badge.svg)](https://github.com/TamerOnLine/repo-fastapi/actions/workflows/ci-macos.yml) |
| **Code Style**| ![Ruff](https://img.shields.io/badge/style-Ruff-000?logo=ruff&logoColor=white) |
| **Tests**     | ![Tests](https://img.shields.io/badge/tests-passing-brightgreen) |
| **Docs**      | [![Docs](https://img.shields.io/badge/docs-available-blue)](docs/API.md) |
| **OS**        | ![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?logo=ubuntu&logoColor=white) ![Windows](https://img.shields.io/badge/Windows-0078D6?logo=windows&logoColor=white) ![macOS](https://img.shields.io/badge/macOS-000000?logo=apple&logoColor=white) |
| **Version**   | [![GitHub release](https://img.shields.io/github/v/release/TamerOnLine/repo-fastapi)](https://github.com/TamerOnLine/repo-fastapi/releases) |
| **License**   | ![License](https://img.shields.io/badge/License-MIT-green) |
| **Support**   | [![Sponsor](https://img.shields.io/badge/Sponsor-💖-pink)](https://paypal.me/tameronline) |
| **GitHub**    | [![Stars](https://img.shields.io/github/stars/TamerOnLine/repo-fastapi?style=social)](https://github.com/TamerOnLine/repo-fastapi/stargazers) [![Forks](https://img.shields.io/github/forks/TamerOnLine/repo-fastapi?style=social)](https://github.co)


---

## 📖 Overview

**NeuroServe** is an **AI Inference Server** built on **FastAPI**, designed to run seamlessly on **GPU (CUDA/ROCm)**, **CPU**, and **macOS MPS**.
It provides ready-to-use REST APIs, a modular **plugin system**, runtime utilities, and a consistent unified response format — making it the perfect foundation for AI-powered services.

---

## Quick Setup
 🔧 Virtualenv quick guide: see **[docs/README_venv.md](docs/README_venv.md)**.

---

## 📚 API Documentation
Detailed API reference and usage examples are available here:
➡️ [API Documentation](docs/API.md)

---

## ✨ Key Features

- 🌐 **REST APIs out-of-the-box** with Swagger UI (`/docs`) & ReDoc (`/redoc`).
- ⚡ **PyTorch integration** with automatic device selection (`cuda`, `cpu`, `mps`, `rocm`).
- 🔌 **Plugin system** to extend functionality with custom AI models or services.
- 📊 **Runtime tools** for GPU info, warm-up routines, and environment inspection.
- 🧠 **Built-in utilities** like a toy model and model size calculator.
- 🧱 **Unified JSON responses** for predictable API behavior.
- 🧪 **Cross-platform CI/CD** (Ubuntu, Windows, macOS, Self-hosted GPU).

---

## 📂 Project Structure

```text
repo-fastapi/
├─ app/                             # application package
│  ├─ core/                         # settings & configuration
│  │  └─ config.py                  # app settings (Pydantic v2)
│  ├─ routes/                       # HTTP API routes
│  ├─ plugins/                      # extensions / integrations
│  ├─ workflows/                    # workflow definitions & orchestrators
│  └─ templates/                    # Jinja templates (if used)
├─ docs/                            # documentation & generated diagrams
│  ├─ ARCHITECTURE.md               # main architecture report
│  ├─ architecture.mmd              # Mermaid source (no fences)
│  ├─ architecture.html             # browser-friendly diagram
│  ├─ architecture.png              # exported PNG (if mmdc installed)
│  ├─ runtime.mmd                   # runtime/infra diagram
│  ├─ imports.mmd                   # Python import graph (if generated)
│  ├─ endpoints.md                  # discovered API endpoints (if generated)
│  └─ README_venv.md                # virtualenv quick guide
├─ tools/                           # project tooling & scripts
│  └─ build_workflows_index.py      # builds docs/workflows-overview.md
├─ tests/                           # test suite
│  └─ test_run.py                   # smoke test for app startup
├─ gen_arch.py                      # architecture generator script
├─ requirements.txt                 # runtime dependencies
├─ requirements-dev.txt             # dev dependencies (ruff, pre-commit, pytest, ...)
├─ .pre-commit-config.yaml          # pre-commit hooks configuration
├─ README.md                        # project overview & usage
└─ LICENSE                          # project license

```

---

## 🏗️ Architecture
For a deeper look into the internal design, modules, and flow of the system, see:
➡️ [Architecture Guide](docs/ARCHITECTURE.md)

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/USERNAME/gpu-server.git
cd gpu-server
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Auto-install PyTorch
```bash
python -m scripts.install_torch --gpu    # or --cpu / --rocm
```

---

## 🚀 Running the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Available endpoints:
- 🏠 **Home** → [http://localhost:8000/](http://localhost:8000/)
- ❤️ **Health** → [http://localhost:8000/health](http://localhost:8000/health)
- 📚 **Swagger UI** → [http://localhost:8000/docs](http://localhost:8000/docs)
- 📘 **ReDoc** → [http://localhost:8000/redoc](http://localhost:8000/redoc)
- 🧭 **Env Summary** → [http://localhost:8000/env](http://localhost:8000/env)
- 🔌 **Plugins** → [http://localhost:8000/plugins](http://localhost:8000/plugins)

Quick test:
```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## 🔌 Plugin System

Each plugin lives in `app/plugins/<name>/` and typically includes:

```
manifest.json
plugin.py        # Defines Plugin class inheriting AIPlugin
README.md        # Documentation
```

API Endpoints:
- `GET /plugins` — list all plugins with metadata.
- `POST /plugins/{name}/{task}` — execute a task inside a plugin.

Example:
```python
from app.plugins.base import AIPlugin

class Plugin(AIPlugin):
    name = "my_plugin"
    tasks = ["infer"]

    def load(self):
        # Load models/resources once
        ...

    def infer(self, payload: dict) -> dict:
        return {"message": "ok", "payload": payload}
```

---

## Workflow System
A lightweight orchestration layer to chain plugins into reproducible pipelines (steps → plugin + task + payload).
All endpoints are exposed under `/workflow`.

- **Endpoints:** `GET /workflow/ping`, `GET /workflow/presets`, `POST /workflow/run`
- **System Guide (EN):** [app/workflows/README.md](app/workflows/README.md)
- **Workflows Index:** [docs/workflows-overview.md](docs/workflows-overview.md)

---

## 🔄 Available Workflows

A full list of available workflows with their versions, tags, and step counts is maintained in the **Workflows Index**.

➡️ [View Workflows Index](docs/workflows-overview.md)

---

## 🧩 Available Plugins

A full list of available plugins with their providers, tasks, and source files is maintained in the **Plugins Index**.

➡️ [View Plugins Index](docs/plugins-overview.md)

---

## 🧪 Development

Install dev dependencies:
```bash
pip install -r requirements-dev.txt
pre-commit install
```

Run tests:
```bash
pytest
```

Ruff (lint + format check) runs automatically via pre-commit hooks.

---

## 🧹 Code Style

We enforce a clean and consistent code style using **Ruff** (linter, import sorter, and formatter).
For full details on configuration, commands, helper scripts, and CI integration, see:

➡️ [Code Style & Linting Guide](docs/CODE_STYLE_GUIDE.md)

---

## 📦 Model Management

Download models in advance:
```bash
python -m scripts.prefetch_models
```

Models are cached in `models_cache/` (see `docs/LICENSES.md` for licenses).

---

## 🏭 Deployment Notes

- Use `uvicorn`/`hypercorn` behind a reverse proxy (e.g., Nginx).
- Configure environment with `APP_*` variables instead of hardcoding.
- Enable HTTPS and configure CORS carefully in production.

---

## 📝 Changelog
A complete history of changes and improvements:
➡️ [CHANGELOG](docs/CHANGELOG.md)

## 📦 Release Notes
Details about the initial release v0.1.0:
➡️ [Release Notes v0.1.0](docs/RELEASE_NOTES_v0.1.0.md)

---

## 🗺️ Roadmap

- [ ] Add `/cuda` endpoint → return detailed CUDA info.
- [ ] Add `/warmup` endpoint for GPU readiness.
- [ ] Provide a **plugin generator CLI**.
- [ ] Implement API Key / JWT authentication.
- [ ] Example plugins: translation, summarization, image classification.
- [ ] Docker support for one-click deployment.
- [ ] Benchmark suite for model inference speed.

---

## 🤝 Contributing

Contributions are welcome!
- Open **Issues** for bugs or ideas.
- Submit **Pull Requests** for improvements.
- Follow style guidelines (Ruff + pre-commit).

---

## 📜 License
Licensed under the **MIT License** — see [LICENSE](./LICENSE).

### 📜 Model Licenses
Some AI/ML models are licensed separately — see [Model Licenses](docs/LICENSES.md).

---
