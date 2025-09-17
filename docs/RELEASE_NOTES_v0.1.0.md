# v0.1.0 — NeuroServe: First Public Preview

**Release date:** 2025-09-13

NeuroServe is now ready as a **reusable FastAPI AI inference server core**.
This release focuses on a clean architecture, a modular plugin system, and cross‑platform readiness (GPU/CPU/MPS/ROCm).

## Highlights
- **Clean architecture**: core config/logging/errors, runtime device layer, unified responses.
- **Plugin system**: build & load model/services as plugins; ships with example `dummy` plugin.
- **Cross‑platform**: CUDA (Linux/Windows), CPU everywhere, MPS on macOS.
- **CI**: Ubuntu, Windows, macOS + self‑hosted GPU lane.
- **Docs**: Updated README with badges table, API guide, and full architecture doc.

## Install & Run
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Test quickly:
```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

## Next up
- API‑Key/JWT auth
- Docker image + compose
- Plugin generator CLI
- Observability (metrics/tracing)
- Example production plugins (ASR / Summarize / Vision)

— thanks for trying NeuroServe!
