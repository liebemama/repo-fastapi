from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import torch
import torchaudio
from transformers import AutoProcessor, WhisperForConditionalGeneration

from app.core.config import get_settings
from app.plugins.base import AIPlugin
from app.runtime import pick_device, pick_dtype
from app.runtime.model_pool import get_model_pool


def _to_mono_tensor(w: torch.Tensor) -> torch.Tensor:
    """Ensure mono layout shaped as (1, T)."""
    if w.dim() == 2 and w.size(0) > 1:
        return w.mean(dim=0, keepdim=True)
    if w.dim() == 1:
        return w.unsqueeze(0)
    return w


class Plugin(AIPlugin):
    tasks = ["speech-to-text"]

    def load(self) -> None:
        """
        Prepare plugin (device, processor, concurrency control), but DO NOT
        load the heavy model here. The model is loaded lazily via ModelPool
        on first use.
        """
        self.settings = get_settings()
        self.dev = pick_device()
        self.model_name = "openai/whisper-small"
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        self._dtype = pick_dtype(str(self.dev))
        self._model_key = f"whisper::{self.model_name}::{self.dev}"
        self._sem = threading.Semaphore(self.settings.MAX_CONCURRENCY_PER_PLUGIN)

        print("[plugin] whisper ready (lazy load) on", self.dev)

    def _factory(self):
        """
        Create and return a ready-to-use model (moved to device, eval, proper dtype).
        Called by ModelPool only when the model is not loaded yet.
        """
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True

        model = (
            WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True,
                dtype=self._dtype,
            )
            .to(self.dev)
            .eval()
        )
        return model

    def _get_model(self):
        """
        Get model from the global pool (loads on first call only).
        """
        pool = get_model_pool()
        return pool.get(self._model_key, self._factory)

    def _download_audio(self, url: str, dst: Path, timeout: int = 30) -> None:
        """Download audio content from the specified URL to the given destination path."""
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    def _load_audio_16k_mono(self, path: Path, max_seconds: int = 600) -> tuple[torch.Tensor, int]:
        """
        Load audio file and return waveform as 1D float32 tensor in [-1..1], sampled at 16kHz.

        Args:
            path (Path): Path to the audio file.
            max_seconds (int): Maximum length of audio in seconds.

        Returns:
            tuple[torch.Tensor, int]: A tuple containing the mono waveform and sample rate.
        """
        waveform: torch.Tensor | None = None
        sr: int | None = None

        try:
            waveform, sr = torchaudio.load(str(path))
        except Exception:
            pass

        if waveform is None:
            try:
                import soundfile as sf

                data, sr = sf.read(str(path), dtype="float32", always_2d=True)
                waveform = torch.from_numpy(data).permute(1, 0)
            except Exception:
                pass

        if waveform is None:
            import wave

            try:
                with wave.open(str(path), "rb") as wf:
                    sr = wf.getframerate()
                    n = wf.getnframes()
                    ch = wf.getnchannels()
                    sw = wf.getsampwidth()
                    raw = wf.readframes(n)
            except Exception as err3:
                raise RuntimeError(f"Failed to load audio file: {path}") from err3

            if sw != 2:
                raise RuntimeError("Unsupported WAV sample width; please use 16-bit PCM.")

            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if ch > 1:
                audio = audio.reshape(-1, ch).mean(axis=1)
            waveform = torch.from_numpy(audio).unsqueeze(0)

        if waveform is None or waveform.numel() == 0:
            raise RuntimeError("Empty or unreadable audio file.")

        waveform = _to_mono_tensor(waveform)

        if waveform.dtype != torch.float32:
            if waveform.dtype == torch.int16:
                waveform = (waveform.to(torch.float32) / 32768.0).clamp_(-1.0, 1.0)
            elif waveform.dtype == torch.int32:
                waveform = (waveform.to(torch.float32) / 2147483648.0).clamp_(-1.0, 1.0)
            else:
                waveform = waveform.to(torch.float32)

        waveform = waveform - waveform.mean(dim=-1, keepdim=True)

        if sr is None:
            raise RuntimeError("Sample rate could not be determined.")

        if max_seconds and max_seconds > 0:
            max_len = int(sr * max_seconds)
            if waveform.size(-1) > max_len:
                waveform = waveform[..., :max_len]

        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000

        waveform = waveform.clamp_(-1.0, 1.0).squeeze(0).contiguous()
        if not torch.isfinite(waveform).all():
            waveform = torch.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)

        return waveform, sr

    def infer(self, payload: dict) -> dict:
        """
        Perform speech-to-text inference using the Whisper model.

        Args:
            payload (dict): Dictionary containing 'audio_url' or 'input', and optional parameters.

        Returns:
            dict: A dictionary with transcription output or error details.
        """
        with self._sem:
            audio_ref = payload.get("audio_url") or payload.get("input")
            if not audio_ref:
                return {"task": "speech-to-text", "error": "audio_url (or input) is required"}

            if isinstance(audio_ref, str):
                p = Path(audio_ref)
                parsed = urlparse(audio_ref)
                if not p.exists() and parsed.scheme not in ("http", "https"):
                    return {
                        "task": "speech-to-text",
                        "error": f"Expected audio file path or URL, got plain text: {audio_ref}",
                    }

            lang = payload.get("language")
            max_new = int(payload.get("max_new_tokens", 256))
            max_new = max(8, min(max_new, 1024))
            tmp_path = Path("tmp_audio.wav")

            try:
                parsed = urlparse(str(audio_ref))
                if parsed.scheme in ["http", "https"]:
                    self._download_audio(str(audio_ref), tmp_path)
                    audio_path = tmp_path
                else:
                    audio_path = Path(str(audio_ref))
                    if not audio_path.exists():
                        return {"task": "speech-to-text", "error": f"Local file not found: {audio_ref}"}

                mono, sr = self._load_audio_16k_mono(audio_path)
                model = self._get_model()

                inputs = self.processor(audio=mono.numpy(), sampling_rate=sr, return_tensors="pt")
                inputs = {
                    k: (
                        v.to(self.dev, dtype=model.dtype)
                        if torch.is_tensor(v) and v.is_floating_point()
                        else (v.to(self.dev) if torch.is_tensor(v) else v)
                    )
                    for k, v in inputs.items()
                }

                limit = getattr(model.config, "max_target_positions", 448)

                if lang:
                    try:
                        prompt_ids = self.processor.get_decoder_prompt_ids(language=lang, task="transcribe")
                        prompt_len = len(prompt_ids) if prompt_ids is not None else 0
                    except Exception:
                        prompt_len = 0
                else:
                    prompt_len = 2

                allowed_new = max(1, limit - prompt_len)
                if max_new > allowed_new:
                    max_new = allowed_new

                if getattr(self.dev, "type", "") == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()

                gen_kwargs = {"max_new_tokens": max_new}
                if lang:
                    gen_kwargs.update({"language": lang, "task": "transcribe"})

                with torch.no_grad():
                    out_ids = model.generate(**inputs, **gen_kwargs)

                if getattr(self.dev, "type", "") == "cuda":
                    torch.cuda.synchronize()
                elapsed = round(time.time() - t0, 3)

                text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]

                return {
                    "task": "speech-to-text",
                    "device": str(self.dev),
                    "model": self.model_name,
                    "language": lang or "auto",
                    "audio_ref": str(audio_ref),
                    "text": text,
                    "elapsed_sec": elapsed,
                }

            except Exception as e:
                return {
                    "task": "speech-to-text",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
