from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.plugins.loader import get_plugin_instance


@dataclass
class StepResult:
    name: str
    ok: bool
    output: dict | None
    error: str | None
    elapsed_sec: float


@dataclass
class StepSpec:
    """تعريف خطوة واحدة: نداء بلجن محدد بمهمة محددة."""

    name: str  # اسم منطقي للخطوة
    plugin: str  # اسم البلجن (كما في نظامك: whisper, dummy, ...)
    task: str  # اسم المهمة داخل البلجن (مثلاً: "speech-to-text")
    payload: dict[str, Any]  # الحمولة المرسلة للبلجن
    timeout: float = 30.0  # مهلة لكل خطوة
    retries: int = 0  # عدد محاولات الإعادة عند الفشل
    retry_backoff: float = 0.0  # تأخير بسيط بين المحاولات (ثواني)


@dataclass
class ParallelBlock:
    """مجموعة خطوات تعمل بالتوازي؛ نعيد كل النتائج لمرحلة لاحقة (rerank/merge)."""

    name: str
    steps: list[StepSpec] = field(default_factory=list)
    timeout: float = 60.0


@dataclass
class WorkflowSpec:
    """تعريف كامل لخط سير عمل."""

    name: str
    sequence: list[StepSpec | ParallelBlock]  # تسلسل: خطوات فردية أو بلوكات متوازية
    rerank_fn: Callable[[list[StepResult]], StepResult] | None = None


class Orchestrator:
    """مشغّل مسارات العمل."""

    async def _run_one(self, step: StepSpec) -> StepResult:
        t0 = time.time()
        last_err: str | None = None
        for attempt in range(step.retries + 1):
            try:
                plugin = get_plugin_instance(step.plugin)
                # كل البلجنات عندك واجهتها sync infer، فنشغّلها بخيط event-loop مخصّص
                output = await asyncio.wait_for(
                    asyncio.to_thread(plugin.infer, dict(step.payload)),
                    timeout=step.timeout,
                )
                # نعتبر ok إذا رجّع dict وفيه لا يوجد مفتاح error
                ok = isinstance(output, dict) and not output.get("error")
                return StepResult(
                    name=step.name,
                    ok=ok,
                    output=output if ok else None,
                    error=None if ok else (output.get("error") if isinstance(output, dict) else "unknown error"),
                    elapsed_sec=round(time.time() - t0, 3),
                )
            except Exception as e:  # noqa: BLE001 (نحتاج تجميع الاستثناء)
                last_err = str(e)
                if attempt < step.retries and step.retry_backoff > 0:
                    await asyncio.sleep(step.retry_backoff)
        # فشل بعد كل المحاولات
        return StepResult(
            name=step.name,
            ok=False,
            output=None,
            error=last_err or "step failed",
            elapsed_sec=round(time.time() - t0, 3),
        )

    async def _run_parallel(self, block: ParallelBlock) -> list[StepResult]:
        async def _runner(s: StepSpec) -> StepResult:
            return await self._run_one(s)

        tasks = [asyncio.create_task(_runner(s)) for s in block.steps]
        done, pending = await asyncio.wait(tasks, timeout=block.timeout)
        # إلغاء أي مهام متبقية
        for p in pending:
            p.cancel()

        results: list[StepResult] = []
        for d in done:
            try:
                results.append(d.result())
            except Exception as e:  # noqa: BLE001
                results.append(StepResult(name="__parallel__", ok=False, output=None, error=str(e), elapsed_sec=0.0))
        return results

    async def run(self, spec: WorkflowSpec) -> dict:
        """يشغّل التسلسل ويُرجع تقريرًا موحّدًا."""
        full_report: list[dict] = []
        carry: dict[str, Any] = {}  # مساحة عمل لمشاركة بيانات بين الخطوات

        for block in spec.sequence:
            if isinstance(block, StepSpec):
                # حقن carry إلى الـ payload (بدون أن نكسر المعطيات الأصلية)
                payload = dict(block.payload)
                if carry:
                    payload.setdefault("_context", {}).update(carry)
                step = StepSpec(**{**block.__dict__, "payload": payload})
                res = await self._run_one(step)
                full_report.append(self._report_step(res))
                # إن نجحت الخطوة نُحدّث carry ببعض المفاتيح الشائعة
                if res.ok and isinstance(res.output, dict):
                    carry.update(
                        {
                            step.name: res.output,
                            "last_text": res.output.get("text") or res.output.get("raw_text"),
                        }
                    )
                else:
                    # لا نوقف المسار افتراضيًا؛ القرار يعود للتصميم. ممكن تضيف خيار "stop_on_fail".
                    pass

            elif isinstance(block, ParallelBlock):
                # حقن carry لكل فرع
                par_steps: list[StepSpec] = []
                for s in block.steps:
                    payload = dict(s.payload)
                    if carry:
                        payload.setdefault("_context", {}).update(carry)
                    par_steps.append(StepSpec(**{**s.__dict__, "payload": payload}))

                results = await self._run_parallel(
                    ParallelBlock(name=block.name, steps=par_steps, timeout=block.timeout)
                )
                full_report.append(
                    {
                        "type": "parallel",
                        "name": block.name,
                        "results": [self._report_step(r) for r in results],
                    }
                )

                # اختيار أفضل نتيجة إن توفر rerank_fn وإلا نأخذ أول OK
                picked: StepResult | None = None
                oks = [r for r in results if r.ok]
                if oks:
                    picked = spec.rerank_fn(oks) if spec.rerank_fn else oks[0]
                    carry.update(
                        {
                            block.name: picked.output,
                            "last_text": (picked.output or {}).get("text") if picked.output else None,
                        }
                    )

        return {
            "workflow": spec.name,
            "report": full_report,
            "context": carry,
        }

    @staticmethod
    def _report_step(r: StepResult) -> dict:
        return {
            "type": "step",
            "name": r.name,
            "ok": r.ok,
            "elapsed_sec": r.elapsed_sec,
            "error": r.error,
            "output_preview": (r.output if isinstance(r.output, dict) else None),
        }


# ---------- جاهز: Presets مفيدة ----------


def rerank_by_longest_text(results: list[StepResult]) -> StepResult:
    """اختيار النتيجة ذات النص الأطول (بديل سريع لحين إضافة RAG/LLM reranker)."""

    def _len(r: StepResult) -> int:
        if r.output and isinstance(r.output, dict):
            t = r.output.get("text") or r.output.get("raw_text") or ""
            return len(str(t))
        return 0

    return max(results, key=_len)


def preset_asr_arabic_pro(audio_url: str) -> WorkflowSpec:
    """
    مسار جاهز:
    1) Whisper-small: تفريغ + postprocess عربي
    2) Parallel: تفريغ بسيط آخر بإعدادات مختلفة (مثلاً max_new مختلفة) لاختيار الأفضل
    3) خطوة تنميق نهائي (باستخدام dummy أو بلجن إعادة صياغة عندك)
    """
    s1 = StepSpec(
        name="whisper_main",
        plugin="whisper",
        task="speech-to-text",
        payload={"audio_url": audio_url, "language": "ar", "postprocess": True},
        timeout=45.0,
        retries=0,
    )

    pblock = ParallelBlock(
        name="asr_candidates",
        steps=[
            StepSpec(
                name="whisper_alt_a",
                plugin="whisper",
                task="speech-to-text",
                payload={"audio_url": audio_url, "language": "ar", "postprocess": True, "max_new_tokens": 196},
                timeout=45.0,
            ),
            StepSpec(
                name="whisper_alt_b",
                plugin="whisper",
                task="speech-to-text",
                payload={"audio_url": audio_url, "language": "ar", "postprocess": True, "max_new_tokens": 320},
                timeout=45.0,
            ),
        ],
        timeout=60.0,
    )

    # خطوة تنميق نهائي (استخدم أي بلجن لديك يوفر "rewrite" أو "polish")
    s3 = StepSpec(
        name="final_polish",
        plugin="dummy",  # غيّرها لبلجنك الحقيقي حين يتوفر
        task="ping",  # مثال: dummy سيعيد payload نفسه؛ استبدله لاحقًا بـ rewrite
        payload={"text": "${_context.last_text}", "note": "polish me"},
        timeout=10.0,
    )

    return WorkflowSpec(
        name="asr_arabic_pro",
        sequence=[s1, pblock, s3],
        rerank_fn=rerank_by_longest_text,
    )
