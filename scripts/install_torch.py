"""
Cross-platform PyTorch installer:
- NVIDIA -> cu124 wheels
- AMD ROCm -> rocm wheels (rocm6.0 by default)
- macOS -> default (CPU wheel with MPS support inside PyTorch)
- Fallback -> CPU
Usage:
  py -m scripts.install_torch --gpu | --cpu | --rocm
Env:
  DEVICE=cuda:0 -> prefer GPU
  DEVICE=cpu    -> force CPU
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def has_nvidia() -> bool:
    try:
        if not have("nvidia-smi"):
            return False
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=5)
        return bool(out.strip())
    except Exception:
        return False


def has_rocm() -> bool:
    # Heuristics: rocminfo or hipcc presence
    return have("rocminfo") or have("hipcc")


def pip_install(args: list[str]) -> int:
    print("\n$ " + " ".join([sys.executable, "-m", "pip"] + args))
    return subprocess.call([sys.executable, "-m", "pip"] + args)


def decide_channel(force_gpu: bool | None, force_cpu: bool | None, force_rocm: bool | None) -> tuple[str, list[str]]:
    # pkgs common
    pkgs = ["torch", "torchvision", "torchaudio"]

    # explicit flags first
    if force_cpu:
        return "CPU (PyPI)", ["install"] + pkgs
    if force_rocm:
        return "ROCm (rocm6.0)", ["install"] + pkgs + ["--index-url", "https://download.pytorch.org/whl/rocm6.0"]
    if force_gpu:
        return "GPU/cu124", ["install"] + pkgs + ["--index-url", "https://download.pytorch.org/whl/cu124"]

    # env hint
    env_device = (os.getenv("DEVICE") or "").lower().strip()
    if env_device.startswith("cuda"):
        return "GPU/cu124", ["install"] + pkgs + ["--index-url", "https://download.pytorch.org/whl/cu124"]
    if env_device == "cpu":
        return "CPU (PyPI)", ["install"] + pkgs

    # auto-detect
    sysname = platform.system().lower()
    if sysname in ("linux", "windows"):
        if has_nvidia():
            return "GPU/cu124", ["install"] + pkgs + ["--index-url", "https://download.pytorch.org/whl/cu124"]
        if has_rocm():
            return "ROCm (rocm6.0)", ["install"] + pkgs + ["--index-url", "https://download.pytorch.org/whl/rocm6.0"]
        return "CPU (PyPI)", ["install"] + pkgs
    # macOS: default wheel (MPS is inside torch for mac)
    return "macOS default (CPU wheel, MPS inside)", ["install"] + pkgs


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--gpu", action="store_true", help="Force NVIDIA CUDA (cu124)")
    g.add_argument("--cpu", action="store_true", help="Force CPU")
    g.add_argument("--rocm", action="store_true", help="Force AMD ROCm (rocm6.0)")
    ap.add_argument("--extra", nargs="*", default=[], help="Extra pip args")
    args = ap.parse_args()

    # already installed?
    try:
        import importlib

        torch = importlib.import_module("torch")
        print(f"PyTorch already installed: {torch.__version__}")
        try:
            print("CUDA available?", getattr(torch, "cuda", None) and torch.cuda.is_available())
            if hasattr(torch.backends, "mps"):
                print("MPS available?", torch.backends.mps.is_available())
        except Exception:
            pass
        return 0
    except Exception:
        pass

    channel, pip_args = decide_channel(args.gpu, args.cpu, args.rocm)
    if args.extra:
        pip_args += args.extra

    print(f"Installing PyTorch [{channel}] ...")
    code = pip_install(pip_args)
    if code != 0:
        print("❌ pip install failed.")
        pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
        print(f"Python detected: {pyver}. If wheels are missing, try Python 3.12 venv.")
        return code

    try:
        import torch

        print(f"✅ Installed torch {torch.__version__}")
        try:
            print("CUDA available?", torch.cuda.is_available())
            if hasattr(torch.backends, "mps"):
                print("MPS available?", torch.backends.mps.is_available())
        except Exception:
            pass
        return 0
    except Exception as e:
        print("⚠️ Installed but cannot import torch:", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
