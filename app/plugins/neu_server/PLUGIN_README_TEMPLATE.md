# 📦 Plugin Name

Short description of what this plugin does.

---

## 📝 Metadata
- **Name:** `plugin-name`
- **Version:** `0.1.0`
- **Author:** Your Name
- **Description:** One-line purpose of the plugin

*(This metadata can also be read from `manifest.json`)*

---

## 🚀 Endpoints

### List Plugins
- **GET** `/plugins`
- **Description:** Returns all loaded plugins and metadata.

### Run Task
- **POST** `/plugins/{name}/{task}`
- **Description:** Execute a task provided by this plugin.
- **Parameters:**
  - `name` → plugin name (e.g., `plugin-name`)
  - `task` → available task (e.g., `predict`, `translate`, `ping`…)

---

## 📤 Request Example

```bash
curl -X POST http://127.0.0.1:8000/plugins/plugin-name/task-name      -H "Content-Type: application/json"      -d '{"text":"Hello World"}'
```

---

## 📥 Response Example

```json
{
  "plugin": "plugin-name",
  "result": {
    "task": "task-name",
    "message": "Result message",
    "payload_received": {
      "text": "Hello World"
    }
  }
}
```

---

## 🔑 Authentication
Currently **no authentication** is required.
*(If API keys are added later, mention here.)*

---

## ⚙️ Installation
Put this plugin folder under `app/plugins/` and restart the server:
```bash
uvicorn app.main:app --reload
```

---

## 📄 License
This plugin follows the same license as the main project (MIT/Apache2…).
