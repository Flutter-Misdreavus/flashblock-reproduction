#!/usr/bin/env python3
"""Freeze the GSM8K dataset revision, split hashes and selected IDs.

Does not evaluate or reveal held-out test items: only split sizes and the
seeded 128 test indices are recorded. Train rows 100..115 are mechanism-stage
prompts and are the only rows whose identifiers are recorded explicitly.
"""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

REVISION = "740312add88f781978c0658806c59bc2815b9866"  # openai/gsm8k main, verified from HF download metadata
TRAIN_PARQUET_SHA = "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d"
TEST_PARQUET_SHA = "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import pyarrow.parquet as pq

    d = Path(args.data_dir)
    train_p = d / "main/train-00000-of-00001.parquet"
    test_p = d / "main/test-00000-of-00001.parquet"
    train_sha, test_sha = sha256(train_p), sha256(test_p)
    if train_sha != TRAIN_PARQUET_SHA or test_sha != TEST_PARQUET_SHA:
        raise SystemExit(f"dataset hash mismatch: train={train_sha} test={test_sha}")
    train = pq.read_table(train_p).to_pylist()
    test = pq.read_table(test_p).to_pylist()
    test_ids = sorted(random.Random(42).sample(range(len(test)), 128))
    train_ids = list(range(100, 116))
    if len(set(test_ids)) != 128 or min(test_ids) < 0 or max(test_ids) >= len(test):
        raise SystemExit("invalid sampled test ids")
    payload = {
        "schema_version": 1,
        "dataset": "openai/gsm8k",
        "config": "main",
        "revision": REVISION,
        "source": "read-only cache /data/research/minicache-reproduction-runtime/data/gsm8k",
        "splits": {
            "train": {"count": len(train), "sha256": train_sha,
                      "bytes": train_p.stat().st_size},
            "test": {"count": len(test), "sha256": test_sha,
                     "bytes": test_p.stat().st_size, "evaluated": False},
        },
        "mechanism_train_indices_100_115": train_ids,
        "formal_test_ids_128_seeded": {
            "rule": "sorted(random.Random(42).sample(range(test_count), 128))",
            "seed": 42,
            "ids": test_ids,
        },
        "test_content_recorded_in_git": False,
        "nested_prompts": {
            "template": "<|im_start|>user\\nPlease reason step by step, and put your final answer within \\\\boxed{}.\\n{problem}<|im_end|>\\n<|im_start|>assistant\\n",
            "train_100_101_sha256": None,
        },
    }
    # freeze the two smoke prompts' identity without exposing test data
    smoke = [train[i] for i in (100, 101)]
    payload["nested_prompts"]["train_100_101_sha256"] = hashlib.sha256(
        json.dumps([r["question"] for r in smoke], sort_keys=True).encode()
    ).hexdigest()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, out)
    print(json.dumps({"train": len(train), "test": len(test), "test_ids": len(test_ids),
                      "out": str(out)}))


if __name__ == "__main__":
    main()
