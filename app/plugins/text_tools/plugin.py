# app/plugins/text_tools/plugin.py
from __future__ import annotations

import re
from typing import Any

from app.plugins.base import AIPlugin


def _normalize_arabic(text: str) -> str:
    # تبسيط (تطبيع) سريع: مسافات، همزات، مدود…
    text = re.sub(r"\s+", " ", text).strip()
    # مثال بسيط: تحويل التنوين/الألف المقصورة وغيرها ممكن توسيعها لاحقاً
    text = text.replace("إ", "ا").replace("أ", "ا").replace("آ", "ا")
    return text


def _spellcheck_ar(text: str) -> str:
    # مكان للتكامل مع تدقيق إملائي (qalsadi/pyarabic/gpt-قواعد…)
    # حالياً: يرجع النص كما هو (stub)
    return text


class Plugin(AIPlugin):
    tasks = ["arabic_normalize", "spellcheck_ar"]

    def load(self) -> None:
        self.name = "text_tools"

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        # يختار المهمة حسب الطلب
        task = payload.get("task") or "arabic_normalize"
        text = payload.get("text")

        # دعم source_key (لو انبنى من orchestrator)
        if text is None and "source_key" in payload:
            # orchestrator يفكها قبل الوصول هنا عادة، ولكن هذا احتياط.
            text = payload["source_key"]

        if not isinstance(text, str) or not text.strip():
            return {"task": task, "error": "text is required", "text": text}

        if task == "arabic_normalize":
            out = _normalize_arabic(text)
            return {"task": task, "text": out}
        elif task == "spellcheck_ar":
            out = _spellcheck_ar(text)
            return {"task": task, "text": out}
        return {"task": task, "error": f"Unknown task: {task}"}
