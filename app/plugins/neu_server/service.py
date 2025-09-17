# app/plugins/neu_server/service.py
class Plugin:
    name = "neu_server"
    tasks = ["summarize", "classify"]

    def infer(self, payload: dict, task: str | None = None):
        if task == "summarize":
            text = (payload or {}).get("text", "")
            return {"summary": text[:200]}
        if task == "classify":
            return {"label": "neutral", "confidence": 0.5}
        return {"ok": True, "echo": payload}
