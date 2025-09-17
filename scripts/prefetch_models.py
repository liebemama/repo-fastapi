import os

import torch
from dotenv import load_dotenv


def info(title):
    """Print a formatted title with separators for clarity."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# Load environment variables from .env
load_dotenv()


# ----- Transformers / HF -----
def prefetch_bart():
    """Prefetch BART model and tokenizer, then perform a dummy generation."""
    from transformers import AutoTokenizer, BartForConditionalGeneration

    name = "facebook/bart-large-cnn"
    info(f"Prefetch BART: {name}")
    tok = AutoTokenizer.from_pretrained(name)
    model = BartForConditionalGeneration.from_pretrained(name, low_cpu_mem_usage=True, dtype=torch.float32)
    x = tok("hello world", return_tensors="pt", truncation=True, max_length=32)
    _ = model.generate(**x, max_new_tokens=8)


def prefetch_distilbert():
    """Prefetch DistilBERT model and tokenizer, then perform a dummy forward pass."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = "distilbert-base-uncased-finetuned-sst-2-english"
    info(f"Prefetch DistilBERT: {name}")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, low_cpu_mem_usage=True, dtype=torch.float32)
    x = tok("a simple test", return_tensors="pt", truncation=True, max_length=32)
    _ = model(**x)


def prefetch_mt5():
    """Prefetch mT5 model and tokenizer, then perform a dummy generation."""
    from transformers import AutoTokenizer, MT5ForConditionalGeneration

    name = "google/mt5-small"
    info(f"Prefetch mT5: {name}")
    tok = AutoTokenizer.from_pretrained(name, use_fast=False)
    model = MT5ForConditionalGeneration.from_pretrained(name, low_cpu_mem_usage=True, dtype=torch.float32)
    x = tok(
        "translate arabic to english: مرحبا",
        return_tensors="pt",
        truncation=True,
        max_length=64,
    )
    _ = model.generate(**x, max_new_tokens=8)


def prefetch_tinyllama():
    """Prefetch TinyLlama model and tokenizer, then perform a dummy generation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    info(f"Prefetch TinyLlama: {name}")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, low_cpu_mem_usage=True, dtype=torch.float32)
    x = tok("Hello", return_tensors="pt")
    _ = model.generate(**x, max_new_tokens=4)


def prefetch_whisper():
    """Prefetch Whisper processor and model, then perform a dummy generation with silent audio."""
    import numpy as np
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    name = "openai/whisper-small"
    info(f"Prefetch Whisper: {name}")
    proc = AutoProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(name, low_cpu_mem_usage=True, dtype=torch.float32)
    dummy = np.zeros(16000, dtype="float32")
    x = proc(audio=dummy, sampling_rate=16000, return_tensors="pt")
    _ = model.generate(**x, max_new_tokens=1)


# ----- Torch hub / torchvision -----
def prefetch_resnet18():
    """Prefetch torchvision ResNet-18 weights."""
    info("Prefetch torchvision resnet18 weights")
    from torchvision.models import ResNet18_Weights, resnet18

    _ = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)


if __name__ == "__main__":
    print("HF_HOME =", os.getenv("HF_HOME"))
    print("TORCH_HOME =", os.getenv("TORCH_HOME"))

    prefetch_bart()
    prefetch_distilbert()
    prefetch_mt5()
    prefetch_tinyllama()
    prefetch_whisper()
    prefetch_resnet18()

    print("\nDone — all models have been downloaded to the unified directories.")
