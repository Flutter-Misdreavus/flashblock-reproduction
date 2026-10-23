#!/usr/bin/env python3
"""Write a sanitized environment snapshot (no credentials, no host secrets)."""
import argparse
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PKGS = [
    "torch", "triton", "numpy", "transformers", "tokenizers", "huggingface-hub",
    "safetensors", "sentencepiece", "accelerate", "omegaconf", "einops", "tqdm",
    "termcolor", "ninja", "psutil", "PyYAML", "pyarrow", "flashinfer-python",
    "flash-attn",
]


def out(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"<error {e!r}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--run-commit", required=True)
    args = ap.parse_args()
    import torch

    pkgs = {}
    for name in PKGS:
        try:
            pkgs[name] = md.version(name)
        except md.PackageNotFoundError:
            pkgs[name] = None
    so = None
    try:
        import flash_attn_2_cuda as fac
        so = str(fac.__file__)
    except Exception as e:
        so = f"<unavailable: {e!r}>"
    root = Path(__file__).resolve().parents[1]
    payload = {
        "schema_version": 1,
        "run_id": args.run_id,
        "run_commit": args.run_commit,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": pkgs,
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
            "visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "flash_attn_2_cuda": so,
        "nvcc": out(["/usr/local/cuda/bin/nvcc", "--version"]),
        "cuda_home": os.environ.get("CUDA_HOME"),
        "nvidia_smi": out(["nvidia-smi", "--query-gpu=uuid,index,name,memory.total,memory.used,utilization.gpu",
                           "--format=csv,noheader"]),
        "driver": out(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "cpu_count": os.cpu_count(),
        "disk_free_bytes": shutil.disk_usage(str(root)).free,
        "git": {
            "head": out(["git", "-C", str(root), "rev-parse", "HEAD"]),
            "branch": out(["git", "-C", str(root), "branch", "--show-current"]),
            "status_short": out(["git", "-C", str(root), "status", "--short"]),
            "remotes": out(["git", "-C", str(root), "remote", "-v"]),
        },
        "pip_freeze": out([sys.executable, "-m", "pip", "freeze"]).splitlines(),
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, p)
    print(json.dumps({"out": str(p), "packages": pkgs}))


if __name__ == "__main__":
    main()
