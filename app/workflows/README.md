# 🧠 Workflow System (FastAPI) — English Documentation

This document explains the Workflow orchestration layer built on top of your FastAPI server. A *workflow* is a sequence of steps; each step invokes a **plugin** with a **task** and a small `payload`. The service exposes a compact HTTP API under `/workflow` and can load workflows dynamically from files in `app/workflows`.

> TL;DR: Think of **plugins** as small atomic capabilities (ASR, text tools, etc.), and **workflows** as **pipelines that chain those capabilities** together.

---

### Files at a glance

```text
app/
├─ routes/
│  └─ workflow.py              # Workflow REST API: /workflow/ping, /workflow/presets, /workflow/run
└─ workflows/                  # File-defined workflows (one folder per workflow)
   └─ <workflow_name>/
      ├─ manifest.json         # Metadata (name, version, sequence_file, tags)
      ├─ workflow.json         # Executable DAG (steps, optional "return")
      └─ README.md             # (optional) per-workflow documentation

docs/
└─ workflows-overview.md       # Auto-generated index of all workflows

tools/
└─ build_workflows_index.py    # Generates/refreshes docs/workflows-overview.md
```


---

## 📦 Project Layout (essentials)

- `app/routes/workflow.py` — the Workflow API router. Declares `GET /workflow/ping`, `GET /workflow/presets`, and `POST /workflow/run`.
- `app/workflows/` — holds file-defined workflows. Each workflow has its own folder with a `manifest.json` and a `workflow.json`.
- `docs/workflows-overview.md` — an auto-generated overview of all workflows.
- `tools/build_workflows_index.py` — script that generates/refreshes the overview and per-workflow README files.

> In most setups, audio samples under `samples/` can be served at an HTTP path like `/samples/test.wav`. If your app does not mount the samples directory, use any accessible URL instead (local or remote).

---

## 🔌 Endpoints

- `GET /workflow/ping`
  Health check; returns `{"ok": true}`.

- `GET /workflow/presets`
  Returns the list of available workflow names (from files and/or in-code presets).

- `POST /workflow/run`
  Runs a workflow in one of three ways:
  1) **Explicit `sequence`** — array of steps (each is `{name, plugin, task, payload}`), or
  2) **Named `preset`** — resolves a workflow by name (from files first, then in-code presets), or
  3) **`auto`** — simple heuristic; currently: if `audio_url` is present, run Whisper-based ASR.

  The system supports **placeholders** (e.g., `{audio_url}`, `{asr.text}`) to pass outputs from previous steps into later steps.

---

## 🚀 Usage Examples

### 1) Run a named preset
List available presets:
```bash
curl http://127.0.0.1:8000/workflow/presets
```

Then run a preset (replace the name if needed):
```bash
curl -X POST http://127.0.0.1:8000/workflow/run \
  -H "Content-Type: application/json" \
  -d '{
    "preset": "arabic_asr_clean",
    "inputs": { "audio_url": "http://127.0.0.1:8000/samples/test.wav" }
  }'
```

**PowerShell:**
```powershell
$json = @'
{
  "preset": "arabic_asr_clean",
  "inputs": { "audio_url": "http://127.0.0.1:8000/samples/test.wav" }
}
'@
Invoke-RestMethod -Uri "http://127.0.0.1:8000/workflow/run" -Method Post -ContentType "application/json" -Body $json
```

### 2) Explicit sequence (Arabic ASR → normalize → spellcheck)
```bash
curl -X POST http://127.0.0.1:8000/workflow/run \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": [
      {
        "name": "asr",
        "plugin": "whisper",
        "task": "speech-to-text",
        "payload": {
          "audio_url": "http://127.0.0.1:8000/samples/test.wav",
          "language": "ar"
        }
      },
      {
        "name": "norm",
        "plugin": "text_tools",
        "task": "arabic_normalize",
        "payload": { "text": "{asr.text}" }
      },
      {
        "name": "spell",
        "plugin": "text_tools",
        "task": "spellcheck_ar",
        "payload": { "text": "{norm.text}" }
      }
    ],
    "return": "spell"
  }'
```

### 3) Auto mode
```bash
curl -X POST http://127.0.0.1:8000/workflow/run \
  -H "Content-Type: application/json" \
  -d '{
    "auto": true,
    "audio_url": "http://127.0.0.1:8000/samples/test.wav"
  }'
```
> The current heuristic is minimal: when it sees `audio_url`, it builds and runs a single-step Whisper ASR workflow. You can extend it later.

---

## 🧱 Step Model (what each step looks like)

Each step has:
- `name` (string): a unique label for referencing outputs in placeholders (e.g., `{name.field}`).
- `plugin` (string): the plugin to invoke.
- `task` (string): the plugin task to run.
- `payload` (object): inputs passed to the plugin’s `infer()` method. Placeholders are allowed.
- `timeout` *(optional)*: per-step timeout (non-enforced placeholder in the base version; useful for future extensions).

Example step:
```json
{
  "name": "asr",
  "plugin": "whisper",
  "task": "speech-to-text",
  "payload": { "audio_url": "{audio_url}" },
  "timeout": 180
}
```

---

## 🧩 Placeholders

You can reference:
- **Root inputs** (from the request’s top-level or the `inputs` object), e.g. `{audio_url}`, `{language}`.
- **Previous step outputs** via dot paths, e.g. `{asr.text}` means: take the `text` field from the output of step `asr`.

> In the base implementation, placeholders are replaced when the entire string equals `{...}`. If you need multiple placeholders inside one string, you can extend the function to use a regex replacer.

---

## 🔁 `return` (partial result)

If you only want the output of a specific step, include `"return": "<stepName>"` in the request.
For example:
```json
{
  "preset": "arabic_asr_clean",
  "inputs": { "audio_url": "http://127.0.0.1:8000/samples/test.wav" },
  "return": "spell"
}
```
The response will include just the `spell` step’s output under `result`.

---

## ⚠️ Errors you may see

- **400** — bad request (no sequence/preset/auto, unknown plugin, unsupported task, or invalid return target).
- **404** — not found (e.g., preset name not found, or plugin not found depending on implementation choice).
- **500** — step execution failed (uncaught error inside a plugin).

The router validates the available plugins (if the plugin registry list API is available) before running the sequence, so frontends get clear errors (400/404) instead of generic 500s.

---

## 🛠️ Create a New Workflow (file-based)

Every workflow folder inside `app/workflows/<name>/` should contain:

1) **manifest.json**
```json
{
  "name": "my_workflow",
  "version": "1.0.0",
  "description": "What it does",
  "sequence_file": "workflow.json",
  "tags": ["demo"]
}
```
- `name`: identifier (avoid spaces).
- `sequence_file`: where to read the sequence from (defaults to `workflow.json`).
- Other fields are freeform metadata you can show in docs.

2) **workflow.json**
```json
{
  "sequence": [
    {
      "name": "asr",
      "plugin": "whisper",
      "task": "speech-to-text",
      "payload": { "audio_url": "{audio_url}" }
    }
  ],
  "return": "asr"
}
```
- The sequence is an array of **steps** (see the model above).
- `"return"` is optional; if present, the API will return only that step’s output.

3) *(Optional)* README in the same folder to document the workflow.

The server’s registry scans `app/workflows/*/manifest.json` and loads definitions dynamically.

---

## 🧾 Generate/Refresh Documentation Index

Use the generator script to build a Markdown overview and keep per-workflow READMEs up-to-date:

```bash
python tools/build_workflows_index.py
# or force-rewrite READMEs as well:
python tools/build_workflows_index.py --force-readme
```

- Writes `docs/workflows-overview.md` with a table of available workflows.
- Creates/updates per-workflow README files from a template.

This is useful for CI to keep docs in sync with the repo’s workflows.

---

## 🧪 Quick sanity checks

```bash
# Health
curl http://127.0.0.1:8000/workflow/ping

# Presets
curl http://127.0.0.1:8000/workflow/presets

# Run (preset)
curl -X POST http://127.0.0.1:8000/workflow/run \
  -H "Content-Type: application/json" \
  -d '{
    "preset": "arabic_asr_clean",
    "inputs": { "audio_url": "http://127.0.0.1:8000/samples/test.wav" }
  }'
```

---

## 💡 Tips

- If you add new plugins, make sure their `tasks` attribute (or method) lists supported task names; the router will check and return a clear 400 if unsupported.
- For local files, you can either mount a static route (e.g., `/samples`) or upload the file elsewhere and pass a public URL.
- If you need stricter validation (e.g., reject empty sequences), add a guard before running.
- Extend the **auto** builder to detect other input types (text-only summarization, TTS, etc.).
- To embed multiple placeholders inside a single string, switch to a regex replacement function.

---

Happy building! ✨
