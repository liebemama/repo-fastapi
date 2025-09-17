# neu_server

**Provider:** neu_server
**Tasks:** summarize, classify

## Description
Short description of what this plugin does and when to use it.

## Installation
- No extra dependencies beyond the project’s standard requirements, unless noted here.
- If this plugin needs additional models, weights, or external assets, list them here.

## Usage

### API Overview
- `GET /plugins` — lists all available plugins.
- `GET /plugins/{name}` — returns metadata for this plugin.
- `POST /plugins/{name}/{task}` — runs a task of this plugin.

> Replace `{name}` with this plugin’s folder name and `{task}` with one of the tasks listed above.

### cURL Example
```bash
curl -X POST "http://localhost:8000/plugins/neu_server/summarize" \
     -H "Content-Type: application/json" \
     -d '{"input":"example"}'
```

### Python Example
```python
import requests

resp = requests.post(
    "http://localhost:8000/plugins/neu_server/summarize",
    json={"input": "example"},
    timeout=60,
)
print(resp.json())
```

## Notes
- If the plugin requires environment variables (e.g., HF_HOME, TORCH_HOME, TRANSFORMERS_OFFLINE), document them here.
- Add relevant reference links (model cards, docs) if applicable.
