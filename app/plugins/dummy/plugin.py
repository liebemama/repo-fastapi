from app.plugins.base import AIPlugin


class Plugin(AIPlugin):
    tasks = ["ping"]

    def load(self) -> None:
        print("[plugin] dummy service ready")

    def infer(self, payload: dict) -> dict:
        return {"task": "ping", "message": "✅ Dummy service is working", "payload_received": payload}
