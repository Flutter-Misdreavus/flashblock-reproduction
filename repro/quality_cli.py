#!/usr/bin/env python3
"""Quality-stage CLI for flashblock-20261023-quality.

Frozen fixed-128 GSM8K test comparison, original TraDo-4B BF16, dense vs
official s2/tau2 A/L cache reuse, ordinary EOS, seed42, batch4, max4096.
One engine per process; one group-condition at a time; raw group outputs are
persisted immediately outside Git.  All scientific parameters are read from
experiments/<run_id>/quality-config.json.

Subcommands
  scorer-tests  run the authored exact-rational scorer tests (CPU, no GPU)
  freeze        verify model/dataset hashes, freeze IDs/prompts/manifest (CPU)
  probe         environment/headroom snapshot
  smoke         train-only 2-prompt ordinary-EOS smoke for both conditions
  generate      32 fixed groups x 2 conditions (64 calls), resumable
  summarize     sanitized per-item table, paired EM/delta/bootstrap/discordance
  ledger        measured resource ledger
  commands      command/provenance ledger
  finalize      write run-dir report.json from frozen evidence
"""
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import quality_scorer as qs  # noqa: E402
import efficiency_cli as eff  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "flashblock-20261023-quality"
CONFIG_PATH = ROOT / "experiments" / RUN_ID / "quality-config.json"
RUN_DIR = ROOT / "runs" / RUN_ID
EXP_DIR = ROOT / "experiments" / RUN_ID
MANIFEST_PATH = EXP_DIR / "input-manifest.json"
RAW_DIR = RUN_DIR / "raw"
INPUT_DIR = RUN_DIR / "inputs"
ATTEMPTS_PATH = RUN_DIR / "gpu-attempts.jsonl"
COMMAND_LOG_PATH = RUN_DIR / "command-log.jsonl"
GENERATION_PATH = RUN_DIR / "generation.json"


class BudgetError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# basics
# ---------------------------------------------------------------------------
def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha256(path):
    return eff._sha256(path)


def _atomic_json(path, obj):
    eff._atomic_json(path, obj)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    if not Path(path).exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _git_head():
    return eff._git_head()


def _git_status():
    return eff._git_status()


def _require_clean(stage):
    st = _git_status()
    if st:
        raise RuntimeError(f"{stage}: refusing GPU/execution with dirty worktree:\n{st}")


def _log_command(stage, argv, status="finished", wall_seconds=None, note=None):
    rec = {"schema_version": 1, "stage": stage, "argv": list(argv),
           "git_head": _git_head(), "at": eff._now(), "status": status,
           "wall_seconds": wall_seconds, "note": note}
    _append_jsonl(COMMAND_LOG_PATH, rec)
    return rec


def _record_attempt(stage, status, wall_seconds, started, finished, **extra):
    rec = {"schema_version": 1, "stage": stage, "status": status,
           "wall_seconds": round(float(wall_seconds), 3), "started": started,
           "finished": finished, "git_head": _git_head(), **extra}
    _append_jsonl(ATTEMPTS_PATH, rec)
    return rec


def _gpu_prior_seconds():
    return float(sum(float(a.get("wall_seconds") or 0.0)
                     for a in _read_jsonl(ATTEMPTS_PATH)))


def _budget_state(cfg, current_elapsed=0.0):
    prior = _gpu_prior_seconds() + float(current_elapsed)
    stage_h = prior / 3600.0
    campaign_h = cfg["budget"]["campaign_before_gpu_hours"] + stage_h
    return {"prior_gpu_seconds_including_current": round(prior, 3),
            "stage_gpu_hours": stage_h, "campaign_gpu_hours": campaign_h,
            "stage_limit_gpu_hours": cfg["budget"]["stage_gpu_hours"],
            "campaign_limit_gpu_hours": cfg["budget"]["campaign_limit_gpu_hours"],
            "reserve_gpu_hours": cfg["budget"]["reserve_gpu_hours"]}


def _budget_guard(cfg, current_elapsed=0.0, margin_seconds=120.0):
    st = _budget_state(cfg, current_elapsed)
    stage_cap = cfg["budget"]["stage_gpu_hours"] * 3600.0 - margin_seconds
    campaign_cap = (cfg["budget"]["campaign_limit_gpu_hours"]
                    - cfg["budget"]["reserve_gpu_hours"]) * 3600.0 - margin_seconds
    if (st["prior_gpu_seconds_including_current"] > stage_cap
            or st["campaign_gpu_hours"] * 3600.0 > campaign_cap):
        raise BudgetError(
            "GPU budget guard: measured stage/campaign allocation would exceed "
            f"stage cap or campaign cap-reserve: {json.dumps(st)}")
    return st


def _budget_deadline(cfg, margin_seconds=120.0, base_prior=None):
    """Monotonic deadline from the GPU-command wall charged at command start."""
    if base_prior is None:
        base_prior = _gpu_prior_seconds()
    stage_left = cfg["budget"]["stage_gpu_hours"] * 3600.0 - base_prior
    campaign_left = ((cfg["budget"]["campaign_limit_gpu_hours"]
                      - cfg["budget"]["reserve_gpu_hours"]
                      - cfg["budget"]["campaign_before_gpu_hours"]) * 3600.0
                     - base_prior)
    left = min(stage_left, campaign_left) - margin_seconds
    if left <= 0:
        raise BudgetError(f"no GPU wall budget remains: stage_left={stage_left:.1f}s "
                          f"campaign_left={campaign_left:.1f}s")
    return time.monotonic() + left, {"base_prior_seconds": round(base_prior, 3),
                                     "stage_left_seconds": round(stage_left, 3),
                                     "campaign_left_seconds": round(campaign_left, 3),
                                     "deadline_margin_seconds": margin_seconds}


def _configure_env(cfg):
    c = cfg["controls"]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = c["gpu_uuid"]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRADO_CONSISTENT_SAMPLING_PARAMS"] = c["consistent_sampling_params_env"]
    os.environ["TRADO_WARMUP_MODEL_LEN"] = str(c["warmup_model_len"])
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ.setdefault("OMP_NUM_THREADS", str(c["cpu_threads"]))
    os.environ.setdefault("MKL_NUM_THREADS", str(c["cpu_threads"]))
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
    os.environ["FLASH_ATTN_CUDA_ARCHS"] = "86"
    venv_bin = str(Path(sys.prefix) / "bin")
    if venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")


def _init_torch():
    return eff._init_torch()


def _config_sha():
    return _sha256(CONFIG_PATH)


def _manifest_sha():
    return _sha256(MANIFEST_PATH)


def _scorer_sha():
    return _sha256(HERE / "quality_scorer.py")


# ---------------------------------------------------------------------------
# freeze / manifest
# ---------------------------------------------------------------------------
def _verify_model(cfg):
    root = Path(cfg["model"]["local_path"])
    results = []
    for rel, expected in sorted(cfg["model"]["files_sha256"].items()):
        path = root / rel
        if not path.is_file():
            raise RuntimeError(f"model file missing: {path}")
        t0 = time.time()
        got = _sha256(path)
        ok = got == expected
        results.append({"path": rel, "bytes": path.stat().st_size,
                        "sha256": got, "expected": expected, "verified": ok,
                        "seconds": round(time.time() - t0, 3)})
        print(f"[model] {rel}: {'OK' if ok else 'MISMATCH'} ({got[:12]})",
              flush=True)
        if not ok:
            raise RuntimeError(f"model hash mismatch for {rel}: {got} != {expected}")
    return {"root": str(root), "files": results,
            "all_verified": True,
            "total_bytes": sum(x["bytes"] for x in results)}


def _build_prompts(tokenizer, rows, cfg):
    instruction = cfg["prompt"]["instruction"]
    template = cfg["prompt"]["template"]
    items = []
    for row, index in zip(rows, [r["_index"] for r in rows]):
        question = row["question"]
        answer = row["answer"]
        content = instruction + "\n" + question
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True)
        hard = template.replace("{problem}", question)
        if chat != hard:
            raise RuntimeError(f"chat template mismatch at test index {index}: "
                               f"renderer != frozen hard template")
        ids = [int(x) for x in tokenizer.encode(chat)]
        if tokenizer.pad_token_id is not None and tokenizer.pad_token_id in ids:
            raise RuntimeError(f"pad token leaked into prompt at index {index}")
        if int(tokenizer.mask_token_id) in ids:
            raise RuntimeError(f"mask token leaked into prompt at index {index}")
        if any(x < 0 or x >= len(tokenizer) for x in ids):
            raise RuntimeError(f"out-of-vocab prompt token at index {index}")
        try:
            gold = qs.extract_gold(answer)
        except ValueError as exc:
            raise RuntimeError(f"malformed gold at test index {index}: {exc}")
        items.append({
            "index": int(index),
            "question": question,
            "answer": answer,
            "gold_numeric": qs.canonical_fraction(gold),
            "prompt": chat,
            "token_ids": ids,
            "prompt_tokens": len(ids),
            "prompt_sha256": _sha256_bytes(chat.encode("utf-8")),
            "template_matches_tokenizer": True,
        })
    return items


def _freeze_test_items(cfg, tokenizer=None):
    from transformers import AutoTokenizer
    ds = cfg["dataset"]
    got = _sha256(ds["test_parquet"])
    if got != ds["test_sha256"]:
        raise RuntimeError(f"test parquet sha256 mismatch: {got}")
    got_train = _sha256(ds["train_parquet"])
    if got_train != ds["train_sha256"]:
        raise RuntimeError(f"train parquet sha256 mismatch: {got_train}")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model"]["local_path"], use_fast=True, trust_remote_code=True)
    ids = sorted(random.Random(42).sample(range(ds["test_count"]), 128))
    if ids != ds["ids"]:
        raise RuntimeError("recomputed frozen test IDs differ from config")
    canonical = _sha256_bytes(json.dumps(ids, separators=(",", ":")).encode())
    if canonical != ds["ids_canonical_sha256"]:
        raise RuntimeError("frozen ID canonical hash mismatch")
    if _sha256(ROOT / ds["formal_test_ids_file"]) != ds["formal_test_ids_file_sha256"]:
        raise RuntimeError("formal test ID file hash mismatch")
    rows = eff._load_gsm8k_rows(ds["test_parquet"], ids)
    items = _build_prompts(tokenizer, rows, cfg)
    if len(items) != 128:
        raise RuntimeError(f"expected 128 frozen items, got {len(items)}")
    if [x["index"] for x in items] != ids:
        raise RuntimeError("frozen item order mismatch")
    max_prompt = max(x["prompt_tokens"] for x in items)
    max_new = int(cfg["controls"]["max_new_tokens"])
    max_len = int(cfg["controls"]["max_model_len"])
    if max_prompt + max_new > max_len:
        raise RuntimeError(f"prompt+max_new {max_prompt}+{max_new} > max_model_len {max_len}")
    # Worst group capacity: group of four largest prompts + 4096 each.
    largest4 = sorted((x["prompt_tokens"] for x in items), reverse=True)[:4]
    group_tokens = sum(x + max_new for x in largest4)
    return ids, items, tokenizer, {
        "max_prompt_tokens": max_prompt,
        "max_new_tokens": max_new,
        "max_model_len": max_len,
        "max_prompt_plus_new": max_prompt + max_new,
        "largest_four_prompts": largest4,
        "worst_group_total_tokens": group_tokens,
    }


def cmd_scorer_tests(args):
    import quality_scorer_tests as qst
    result = qst.run_tests()
    result["run_id"] = RUN_ID
    result["created_at"] = eff._now()
    result["git_head"] = _git_head()
    _atomic_json(EXP_DIR / "scorer-test-results.json", result)
    _atomic_json(RUN_DIR / "scorer-test-results.json", result)
    _log_command("scorer_tests", args.argv, status=result["status"])
    print(json.dumps({"status": result["status"], "passed": result["passed"],
                      "failed": result["failed"],
                      "scorer_sha256": result["scorer_sha256"]}, indent=2))
    return 0 if result["status"] == "PASSED" else 1


def cmd_freeze(args):
    _require_clean("freeze")
    cfg = load_config()
    ds = cfg["dataset"]
    if shutil.disk_usage(ROOT).free < 40 * 2**30:
        raise RuntimeError("filesystem free < 40 GiB")
    print("[freeze] verifying model files", flush=True)
    model = _verify_model(cfg)
    print("[freeze] verifying test split, IDs, prompts and gold", flush=True)
    ids, items, tokenizer, capacity = _freeze_test_items(cfg)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_prompts = {"schema_version": 1, "run_id": RUN_ID,
                   "note": "raw test question/prompt text stays outside Git",
                   "items": items}
    _atomic_json(INPUT_DIR / "test-prompts.json", raw_prompts)
    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "parent_run_id": cfg["parent_run_id"],
        "created_at": eff._now(),
        "git_head": _git_head(),
        "config_path": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": _config_sha(),
        "model": {
            "model_id": cfg["model"]["model_id"],
            "revision": cfg["model"]["revision"],
            "local_path": cfg["model"]["local_path"],
            "total_bytes": model["total_bytes"],
            "all_files_sha256_verified": model["all_verified"],
            "config_sha256": cfg["model"]["config_sha256"],
            "chat_template_sha256": cfg["model"]["chat_template_sha256"],
            "files": [{"path": x["path"], "bytes": x["bytes"],
                       "sha256": x["sha256"], "verified": x["verified"]}
                      for x in model["files"]],
        },
        "dataset": {
            "dataset": ds["dataset"], "config": ds["config"],
            "revision": ds["revision"],
            "test_parquet": ds["test_parquet"], "test_sha256": ds["test_sha256"],
            "test_count": ds["test_count"],
            "train_parquet": ds["train_parquet"], "train_sha256": ds["train_sha256"],
            "train_count": ds["train_count"],
            "ids_rule": ds["ids_rule"], "ids_seed": ds["ids_seed"],
            "ids": ids, "ids_canonical_sha256": ds["ids_canonical_sha256"],
            "formal_test_ids_file": ds["formal_test_ids_file"],
            "formal_test_ids_file_sha256": ds["formal_test_ids_file_sha256"],
            "test_content_recorded_in_git": False,
        },
        "template": {
            "instruction": cfg["prompt"]["instruction"],
            "instruction_sha256": _sha256_bytes(
                cfg["prompt"]["instruction"].encode("utf-8")),
            "template": cfg["prompt"]["template"],
            "template_sha256": _sha256_bytes(
                cfg["prompt"]["template"].encode("utf-8")),
            "content_rule": cfg["prompt"]["content_rule"],
            "apply_chat_template": cfg["prompt"]["apply_chat_template"],
            "all_items_match_tokenizer_template": True,
        },
        "scorer": {
            "path": cfg["scorer"]["path"], "sha256": _scorer_sha(),
            "tests_path": cfg["scorer"]["tests_path"],
            "tests_sha256": _sha256(HERE / "quality_scorer_tests.py"),
            "rule": cfg["scorer"]["rule"],
        },
        "controls": cfg["controls"],
        "groups": cfg["groups"],
        "capacity": capacity,
        "items": [{"index": x["index"],
                   "prompt_sha256": x["prompt_sha256"],
                   "prompt_tokens": x["prompt_tokens"],
                   "gold_numeric": x["gold_numeric"]} for x in items],
        "raw_prompts_relpath": f"runs/{RUN_ID}/inputs/test-prompts.json",
        "raw_prompts_sha256": _sha256(INPUT_DIR / "test-prompts.json"),
    }
    _atomic_json(MANIFEST_PATH, manifest)
    _atomic_json(RUN_DIR / "input-manifest.json", manifest)
    shutil.copy2(CONFIG_PATH, RUN_DIR / "quality-config.json")
    _log_command("freeze", args.argv, status="SUCCEEDED")
    print(json.dumps({
        "status": "SUCCEEDED", "manifest": str(MANIFEST_PATH),
        "manifest_sha256": _manifest_sha(),
        "ids_canonical_sha256": ds["ids_canonical_sha256"],
        "max_prompt_tokens": capacity["max_prompt_tokens"],
        "worst_group_total_tokens": capacity["worst_group_total_tokens"],
        "model_total_bytes": model["total_bytes"],
    }, indent=2))
    return 0


# ---------------------------------------------------------------------------
# environment probe
# ---------------------------------------------------------------------------
def cmd_probe(args):
    t0 = time.time()
    started = eff._now()
    cfg = load_config()
    _configure_env(cfg)
    torch = _init_torch()
    probe = eff.NvmlSampler(cfg["controls"]["gpu_uuid"], interval=99)
    probe._init_nvml()
    q0 = probe.query()
    env = {
        "schema_version": 1, "run_id": RUN_ID, "created_at": eff._now(),
        "git_head": _git_head(), "python": sys.version.replace("\n", " "),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": q0,
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "cpu_count": os.cpu_count(),
        "packages": {},
        "nvidia_smi": None,
    }
    try:
        import importlib.metadata as md
        for name in ("torch", "triton", "numpy", "transformers", "tokenizers",
                     "flashinfer-python", "flash-attn", "xxhash", "pyarrow",
                     "pynvml", "matplotlib"):
            try:
                env["packages"][name] = md.version(name)
            except Exception:
                env["packages"][name] = None
    except Exception:
        pass
    try:
        env["nvidia_smi"] = __import__("subprocess").check_output(
            ["nvidia-smi", "--query-gpu=uuid,index,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader"], text=True, timeout=10).strip()
    except Exception as e:
        env["nvidia_smi_error"] = repr(e)
    _atomic_json(EXP_DIR / "environment.json", env)
    _atomic_json(RUN_DIR / "environment.json", env)
    wall = time.time() - t0
    _record_attempt("environment_probe", "SUCCEEDED", wall, started, eff._now())
    _log_command("environment_probe", args.argv, status="SUCCEEDED", wall_seconds=wall)
    print(json.dumps(env, indent=2))
    return 0


# ---------------------------------------------------------------------------
# one group-condition generation
# ---------------------------------------------------------------------------
def _params_finite(module, chunk=1 << 20):
    """Chunked full-parameter finiteness scan with bounded temporaries.

    The earlier whole-parameter `(~isfinite(flat)).sum()` allocated large
    int64 reduction/cast temporaries (retained by the caching allocator), which
    could consume several GiB during validation.  This proven chunked form
    scans 1 Mi elements at a time and never materialises a whole-parameter
    reduction.
    """
    import torch
    total = 0
    nonfinite_tensors = []
    nonfinite_elems = 0
    for name, p in module.named_parameters():
        total += p.numel()
        flat = p.detach().view(-1)
        bad = 0
        for i in range(0, flat.numel(), chunk):
            part = flat[i:i + chunk]
            finite = torch.isfinite(part)
            if not bool(finite.all().item()):
                bad += int((~finite).sum().item())
            del finite, part
        if bad:
            nonfinite_tensors.append({"name": name, "nonfinite": bad})
            nonfinite_elems += bad
    return {"tensors": sum(1 for _ in module.parameters()), "elements": total,
            "nonfinite_elements": nonfinite_elems,
            "nonfinite_tensors": nonfinite_tensors[:20]}


def _install_finite_hook(h, audit):
    import torch
    mr = h["model_runner"]
    orig = mr.run_model

    def wrapped(*a, **k):
        logits = orig(*a, **k)
        audit["calls"] += 1
        audit["max_numel"] = max(audit.get("max_numel", 0), int(logits.numel()))
        if not bool(torch.isfinite(logits).all().item()):
            audit["nonfinite_calls"] += 1
            raise RuntimeError(
                f"nonfinite logits at audited run_model call {audit['calls']}")
        return logits

    mr.run_model = wrapped
    h["orig_run_model"] = orig


def _release_finite_hook(h):
    """Drop the audited-run_model closure before engine teardown."""
    mr = h.get("model_runner")
    orig = h.pop("orig_run_model", None)
    if mr is not None and orig is not None:
        try:
            mr.run_model = orig
        except Exception:
            pass


def _memory_pre_generation(torch):
    """Snapshot true init/validation peaks after chunked finite scanning."""
    import torch as T
    stats = {
        "init_peak_allocated_mib": round(float(T.cuda.max_memory_allocated() / 2**20), 1),
        "init_peak_reserved_mib": round(float(T.cuda.max_memory_reserved() / 2**20), 1),
    }
    return stats


def _finish_memory_scan(torch, stats):
    import torch as T
    stats["scan_peak_allocated_mib"] = round(float(T.cuda.max_memory_allocated() / 2**20), 1)
    stats["scan_peak_reserved_mib"] = round(float(T.cuda.max_memory_reserved() / 2**20), 1)
    stats["pre_generation_peak_allocated_mib"] = max(
        stats["init_peak_allocated_mib"], stats["scan_peak_allocated_mib"])
    stats["pre_generation_peak_reserved_mib"] = max(
        stats["init_peak_reserved_mib"], stats["scan_peak_reserved_mib"])
    T.cuda.empty_cache()
    stats["allocated_after_empty_cache_mib"] = round(
        float(T.cuda.memory_allocated() / 2**20), 1)
    stats["reserved_after_empty_cache_mib"] = round(
        float(T.cuda.memory_reserved() / 2**20), 1)
    return stats


def _run_batch(h, cfg, items, cond, ratio, group_index, order_pos, max_tokens,
               commit):
    _init_torch()
    import torch as T
    from jetengine_ext.sampling_params import SamplingParams
    c = cfg["controls"]
    llm = h["llm"]
    h["identity_enabled"] = False
    eff._reset_engine(h)
    h["finished"] = {}
    h["finished_order"] = []
    h["counters"] = Counter()
    h["per_seq_identity"] = {}
    seedinfo = eff._seed_all(c["seed"])
    sps = [SamplingParams(
        temperature=c["temperature"], max_tokens=int(max_tokens),
        ignore_eos=bool(c["ignore_eos"]), block_length=c["block_length"],
        denoising_steps=c["denoising_steps"],
        dynamic_threshold=c["dynamic_threshold"], topk=c["top_k"],
        topp=c["top_p"], remasking_strategy=c["remasking_strategy"])
        for _ in items]
    T.cuda.synchronize()
    T.cuda.reset_peak_memory_stats()
    load_probe = eff.NvmlSampler(c["gpu_uuid"], interval=99)
    load_probe._init_nvml()
    load_start = load_probe.query()
    seqs = []
    error = None
    tb = None
    t0 = time.perf_counter()
    prefill_seconds = None
    decode_seconds = None

    def checked_step():
        deadline = h.get("budget_deadline")
        if deadline is not None and time.monotonic() > deadline:
            raise BudgetError("live monotonic GPU wall deadline exceeded inside a group")
        eff._step_h(h, ratio)

    try:
        # Sequential prefill, identical both conditions (scheduler does not
        # enforce max_num_batched_tokens).
        for i, item in enumerate(items):
            seq = eff._make_seq(h, item["token_ids"], sps[i], ratio)
            llm.scheduler.add(seq)
            seqs.append(seq)
            guard = 0
            while seq.status.name == "WAITING":
                checked_step()
                guard += 1
                if guard > 8:
                    raise RuntimeError(f"prefill did not complete for {item['index']}")
        T.cuda.synchronize()
        prefill_seconds = time.perf_counter() - t0
        t_dec = time.perf_counter()
        while llm.scheduler.running:
            checked_step()
        T.cuda.synchronize()
        decode_seconds = time.perf_counter() - t_dec
    except BudgetError:
        raise
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
        tb = traceback.format_exc()[-6000:]
        try:
            T.cuda.synchronize()
        except Exception:
            pass
    load_end = load_probe.query()
    peak_alloc = float(T.cuda.max_memory_allocated() / 2**20)
    peak_res = float(T.cuda.max_memory_reserved() / 2**20)
    eos_id = int(getattr(llm.tokenizer, "eos_token_id", -1))
    rows = []
    problems = []
    if peak_alloc > float(c["peak_allocated_limit_mib"]):
        problems.append(
            f"boundary resource violation: torch allocated {peak_alloc:.1f} MiB")
    if float(load_end["mem_free_mib"]) < float(c["min_free_mib"]):
        problems.append(
            f"boundary resource violation: physical free {load_end['mem_free_mib']:.1f} MiB")
    if h.get("sampler") is not None and h["sampler"].breach is not None:
        problems.append(f"sampler guard breach: {h['sampler'].breach}")
    for seq_pos, seq in enumerate(seqs):
        sid = int(seq.seq_id)
        ids = h["finished"].get(sid)
        item = items[seq_pos]
        if ids is None:
            problems.append(f"seq {sid} no finished output")
            continue
        ids = [int(x) for x in ids]
        ids_in_vocab = all(0 <= x < len(llm.tokenizer) for x in ids)
        contains_mask = int(llm.config.mask_token_id) in ids
        eos_present = bool(ids) and ids[-1] == eos_id
        if eos_present:
            finish = "eos"
        elif len(ids) >= int(max_tokens):
            finish = "max_tokens"
        else:
            finish = "other"
        truncated = finish == "max_tokens"
        text = llm.tokenizer.decode(ids, skip_special_tokens=False)
        clean_text = llm.tokenizer.decode(ids, skip_special_tokens=True)
        seq_completion = len(ids)
        try:
            seq_completion = int(seq.num_completion_tokens)
        except Exception:
            pass
        if seq_completion != len(ids):
            problems.append(
                f"seq {sid} raw completion length {len(ids)} != seq {seq_completion}")
        if not ids_in_vocab:
            problems.append(f"seq {sid} out-of-vocab id")
        if contains_mask:
            problems.append(f"seq {sid} mask id in output")
        if len(ids) == 0:
            problems.append(f"seq {sid} empty raw output")
        rows.append({
            "index": int(item["index"]),
            "prompt_sha256": item["prompt_sha256"],
            "prompt_tokens": int(item["prompt_tokens"]),
            "token_ids": ids,
            "text": text,
            "output_sha256": _sha256_bytes(
                json.dumps(ids, separators=(",", ":")).encode()),
            "new_tokens": len(ids),
            "finish_reason": finish,
            "eos_present": eos_present,
            "truncated": truncated,
            "empty_output": not bool(clean_text.strip()),
            "ids_in_vocab": ids_in_vocab,
            "contains_mask_token": contains_mask,
            "seq_id": sid,
        })
    if len(rows) != len(items):
        problems.append(f"collected {len(rows)} rows for {len(items)} items")
    if error:
        problems.append(f"generation exception: {error}")
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "group_index": int(group_index),
        "condition": cond,
        "ratio": float(ratio),
        "group_condition_order": int(order_pos),
        "execution_commit": commit,
        "config_sha256": _config_sha(),
        "manifest_sha256": _manifest_sha() if MANIFEST_PATH.exists() else None,
        "scorer_sha256": _scorer_sha(),
        "eos_token_id": eos_id,
        "seed": int(c["seed"]),
        "seed_before_run": seedinfo,
        "max_new_tokens": int(max_tokens),
        "batch_size": len(items),
        "items": rows,
        "prefill_seconds": round(prefill_seconds, 4) if prefill_seconds is not None else None,
        "decode_seconds": round(decode_seconds, 4) if decode_seconds is not None else None,
        "wall_seconds": round(time.perf_counter() - t0, 3),
        "counters": {k: int(v) for k, v in sorted(h["counters"].items())},
        "peak_allocated_mib": round(peak_alloc, 1),
        "peak_reserved_mib": round(peak_res, 1),
        "load_start": load_start,
        "load_end": load_end,
        "problems": problems,
        "status": "SUCCEEDED" if not problems else "FAILED",
        "error": error,
        "traceback": tb,
    }


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------
def _train_items(cfg, tokenizer, indices):
    ds = cfg["dataset"]
    rows = eff._load_gsm8k_rows(ds["train_parquet"], indices)
    return _build_prompts(tokenizer, rows, cfg)


def cmd_smoke(args):
    _require_clean("smoke")
    cfg = load_config()
    if _manifest_sha() != args.manifest_sha:
        raise RuntimeError("smoke manifest hash does not match frozen manifest")
    execution_commit = _git_head()
    _budget_guard(cfg)
    base_prior = _gpu_prior_seconds()
    deadline, budget_info = _budget_deadline(cfg, base_prior=base_prior)
    _configure_env(cfg)
    torch = _init_torch()
    eff._setup_paths()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["local_path"], use_fast=True, trust_remote_code=True)
    items = _train_items(cfg, tokenizer, [100, 101])
    for item in items:
        item["gold_numeric"] = qs.canonical_fraction(qs.extract_gold(item["answer"]))
    # Preserve any previous failed smoke attempt instead of overwriting it.
    preserved = []
    if (RUN_DIR / "train-smoke.json").exists():
        fail_dir = RUN_DIR / "smoke-attempts"
        fail_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        for name in ("train-smoke.json", "train-smoke-dense.json", "train-smoke-s2.json"):
            src = RUN_DIR / name
            if src.exists():
                dest = fail_dir / f"{src.stem}-attempt-{stamp}{src.suffix}"
                shutil.move(str(src), str(dest))
                preserved.append(str(dest))
    started = eff._now()
    t0 = time.time()
    sampler = eff.NvmlSampler(
        cfg["controls"]["gpu_uuid"], interval=0.5,
        min_free_mib=cfg["resource"]["physical_free_min_mib"],
        max_alloc_mib=cfg["resource"]["own_peak_allocated_mib_limit"])
    sampler.start()
    h = None
    result = {"schema_version": 1, "run_id": RUN_ID, "stage": "train_smoke",
              "scope": "GSM8K train prompts 100,101; ordinary EOS; max 256 new tokens; smoke only, no accuracy claim",
              "execution_commit": execution_commit, "status": "FAILED",
              "config_sha256": _config_sha(), "manifest_sha256": _manifest_sha(),
              "budget_deadline": budget_info,
              "commands": ["smoke train100..101 both conditions ordinary EOS max256"],
              "preserved_previous_attempts": preserved,
              "conditions": {}}
    try:
        free = eff._require_free_headroom(
            cfg, minimum_mib=cfg["resource"]["preconstruction_free_min_mib"],
            label="smoke engine construction")
        result["preconstruction"] = free
        h = {"cfg": cfg, "batch": 2, "llm": eff._make_engine(cfg, 2),
             "identity_layers": [], "expected": {}, "sampler": sampler,
             "budget_deadline": deadline}
        h["model_runner"] = h["llm"].model_runner
        eff._install_recorder(h, [])
        kv = int(h["llm"].config.num_kvcache_blocks)
        cal = len(h["llm"].scheduler.cal_blocks)
        need = 2 * math.ceil(cfg["controls"]["max_model_len"] / 256)
        result["kv_cache"] = {"blocks": kv, "cal_blocks": cal,
                              "free_data_blocks": kv - cal,
                              "needed_data_blocks": need}
        if kv - cal < need:
            raise RuntimeError(f"KV capacity {kv - cal} < {need}")
        memory_pre = _memory_pre_generation(torch)
        result["finite_parameters"] = _params_finite(h["model_runner"].model)
        memory_pre = _finish_memory_scan(torch, memory_pre)
        result["memory_pre_generation"] = memory_pre
        if result["finite_parameters"]["nonfinite_elements"] != 0:
            raise RuntimeError("nonfinite model parameters")
        audit = {"calls": 0, "nonfinite_calls": 0, "max_numel": 0}
        _install_finite_hook(h, audit)
        result["logits_audit"] = audit
        ok_all = True
        for pos, cond in enumerate(["dense", "s2"]):
            ratio = cfg["controls"]["sparsity_ratio_dense"] if cond == "dense" else cfg["controls"]["sparsity_ratio_s2"]
            res = _run_batch(h, cfg, items, cond, ratio, group_index=-1,
                             order_pos=pos, max_tokens=256,
                             commit=execution_commit)
            result["conditions"][cond] = res
            _atomic_json(RUN_DIR / f"train-smoke-{cond}.json", res)
            if res["status"] != "SUCCEEDED":
                ok_all = False
                result["stopped_after_failed_condition"] = cond
                break
            if cond == "s2":
                if int(res["counters"].get("reuse_rows", 0)) <= 0:
                    ok_all = False
                    result["s2_reuse_failure"] = "no reuse rows observed"
                if int(res["counters"].get("update_rows", 0)) <= 0:
                    ok_all = False
                    result["s2_update_failure"] = "no need_update/update rows observed"
            else:
                if int(res["counters"].get("reuse_rows", 0)) != 0:
                    ok_all = False
                    result["dense_reuse_failure"] = "dense unexpectedly reused cache"
        if audit["calls"] <= 0 or audit["nonfinite_calls"] != 0:
            ok_all = False
        if result["finite_parameters"]["nonfinite_elements"] != 0:
            ok_all = False
        if sampler.breach is not None:
            ok_all = False
            result["resource_breach"] = sampler.breach
        result["validation"] = {
            "both_conditions_valid": ok_all,
            "logits_finite": audit["calls"] > 0 and audit["nonfinite_calls"] == 0,
            "parameters_finite": result["finite_parameters"]["nonfinite_elements"] == 0,
            "raw_ids_and_counts_valid": all(
                c["status"] == "SUCCEEDED" for c in result["conditions"].values()),
            "s2_reuse_observed": int(result["conditions"].get("s2", {}).get("counters", {}).get("reuse_rows", 0)) > 0,
            "dense_no_reuse": int(result["conditions"].get("dense", {}).get("counters", {}).get("reuse_rows", 0)) == 0,
        }
        result["status"] = "SUCCEEDED" if all(result["validation"].values()) else "FAILED"
    except Exception as exc:  # noqa: BLE001
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()[-6000:]
    finally:
        result["sampler"] = sampler.summary()
        if h is not None:
            result["engine_allocated_after_release_mib"] = None
            try:
                _release_finite_hook(h)
                eff._release_engine(h)
                torch.cuda.synchronize()
                result["engine_allocated_after_release_mib"] = round(
                    float(torch.cuda.memory_allocated() / 2**20), 1)
            except Exception as exc:  # noqa: BLE001
                result["engine_release_error"] = repr(exc)
        result["wall_seconds"] = round(time.time() - t0, 3)
        result["finished_at"] = eff._now()
        _atomic_json(RUN_DIR / "train-smoke.json", result)
        small = {k: v for k, v in result.items() if k not in ("conditions",)}
        small["conditions"] = {}
        for cond, res in result.get("conditions", {}).items():
            small["conditions"][cond] = {
                k: v for k, v in res.items()
                if k not in ("items", "traceback", "seed_before_run")
            }
            small["conditions"][cond]["items"] = [
                {k: v for k, v in row.items() if k not in ("text", "token_ids")}
                for row in res.get("items", [])
            ]
        _atomic_json(EXP_DIR / "train-smoke.json", small)
        _record_attempt("train_smoke", result["status"], result["wall_seconds"],
                        started, eff._now(),
                        execution_commit=result["execution_commit"])
        _log_command("train_smoke", args.argv, status=result["status"],
                     wall_seconds=result["wall_seconds"])
    print(json.dumps({"status": result["status"],
                      "validation": result.get("validation"),
                      "error": result.get("error")}, indent=2))
    return 0 if result["status"] == "SUCCEEDED" else 1


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def _group_path(group_index, condition):
    return RAW_DIR / f"group-{group_index:02d}-{condition}.json"


def _load_frozen(cfg):
    man = _load_json(MANIFEST_PATH)
    if man["config_sha256"] != _config_sha():
        raise RuntimeError("manifest config hash mismatch")
    raw = _load_json(ROOT / man["raw_prompts_relpath"])
    if _sha256(ROOT / man["raw_prompts_relpath"]) != man["raw_prompts_sha256"]:
        raise RuntimeError("raw prompt file hash mismatch")
    by_index = {int(x["index"]): x for x in raw["items"]}
    if len(by_index) != 128:
        raise RuntimeError("raw prompts do not contain exactly 128 items")
    return man, by_index


def _validate_completed_group(path, man, group_index, condition, cfg):
    if not path.exists():
        return None
    data = _load_json(path)
    if data.get("status") != "SUCCEEDED":
        return None
    if int(data.get("group_index", -1)) != group_index or data.get("condition") != condition:
        raise RuntimeError(f"completed group identity mismatch: {path}")
    if data.get("problems") not in ([], None):
        raise RuntimeError(f"completed group carries problems: {path}")
    if len(data.get("items", [])) != 4:
        raise RuntimeError(f"completed group does not have 4 items: {path}")
    if data.get("config_sha256") != man["config_sha256"]:
        raise RuntimeError(f"completed group config hash mismatch: {path}")
    if data.get("manifest_sha256") != _manifest_sha():
        raise RuntimeError(f"completed group manifest hash mismatch: {path}")
    if data.get("scorer_sha256") != man["scorer"]["sha256"]:
        raise RuntimeError(f"completed group scorer hash mismatch: {path}")
    if not data.get("execution_commit"):
        raise RuntimeError(f"completed group missing execution commit: {path}")
    if int(data.get("seed", -1)) != int(cfg["controls"]["seed"]):
        raise RuntimeError(f"completed group seed mismatch: {path}")
    if int(data.get("max_new_tokens", -1)) != int(cfg["controls"]["max_new_tokens"]):
        raise RuntimeError(f"completed group max-token limit mismatch: {path}")
    if int(data.get("batch_size", -1)) != int(cfg["controls"]["batch_size"]):
        raise RuntimeError(f"completed group batch size mismatch: {path}")
    order = (cfg["groups"]["order_even_group_index"] if group_index % 2 == 0
             else cfg["groups"]["order_odd_group_index"])
    if int(data.get("group_condition_order", -1)) != order.index(condition):
        raise RuntimeError(f"completed group condition order mismatch: {path}")
    eos_id = int(data.get("eos_token_id", -1))
    ids_expected = sorted(int(x["index"]) for x in data["items"])
    expected = man["dataset"]["ids"][group_index * 4:(group_index + 1) * 4]
    if ids_expected != expected:
        raise RuntimeError(f"completed group IDs mismatch: {path}")
    for row in data["items"]:
        meta = next(x for x in man["items"] if int(x["index"]) == int(row["index"]))
        ids = [int(x) for x in row["token_ids"]]
        if row["prompt_sha256"] != meta["prompt_sha256"]:
            raise RuntimeError(f"completed group prompt hash mismatch: {path}")
        if int(row["prompt_tokens"]) != int(meta["prompt_tokens"]):
            raise RuntimeError(f"completed group prompt length mismatch: {path}")
        if int(row["new_tokens"]) != len(ids):
            raise RuntimeError(f"completed group token count mismatch: {path}")
        if row["output_sha256"] != _sha256_bytes(
                json.dumps(ids, separators=(",", ":")).encode()):
            raise RuntimeError(f"completed group output hash mismatch: {path}")
        if bool(row["truncated"]) != (row["finish_reason"] == "max_tokens"):
            raise RuntimeError(f"completed group truncation flag mismatch: {path}")
        if bool(row["eos_present"]) != (bool(ids) and ids[-1] == eos_id):
            raise RuntimeError(f"completed group EOS flag mismatch: {path}")
        if not bool(row["ids_in_vocab"]) or bool(row["contains_mask_token"]):
            raise RuntimeError(f"completed group invalid raw IDs: {path}")
        if len(ids) > int(cfg["controls"]["max_new_tokens"]) or len(ids) < 1:
            raise RuntimeError(f"completed group length outside limit: {path}")
    return data


def cmd_generate(args):
    _require_clean("generate")
    cfg = load_config()
    man, by_index = _load_frozen(cfg)
    ids = man["dataset"]["ids"]
    if len(ids) != 128:
        raise RuntimeError("manifest does not contain 128 frozen IDs")
    _configure_env(cfg)
    torch = _init_torch()
    eff._setup_paths()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    execution_commit = _git_head()
    if cfg["controls"]["ignore_eos"] is not False:
        raise RuntimeError("heldout generation requires ignore_eos=false")
    if int(cfg["controls"]["max_new_tokens"]) != 4096:
        raise RuntimeError("heldout generation requires exact max_new_tokens=4096")
    if int(cfg["controls"]["batch_size"]) != 4 or int(cfg["controls"]["max_active"]) != 4:
        raise RuntimeError("heldout generation requires batch_size=max_active=4")
    _budget_guard(cfg)
    base_prior = _gpu_prior_seconds()
    deadline, budget_info = _budget_deadline(cfg, base_prior=base_prior)
    started = eff._now()
    t_start = time.time()
    sampler = eff.NvmlSampler(
        cfg["controls"]["gpu_uuid"], interval=0.5,
        min_free_mib=cfg["resource"]["physical_free_min_mib"],
        max_alloc_mib=cfg["resource"]["own_peak_allocated_mib_limit"])
    sampler.start()
    h = None
    progress = {"schema_version": 1, "run_id": RUN_ID,
                "execution_commit": execution_commit, "status": "RUNNING",
                "started": started, "groups_total": 32,
                "completed_group_conditions": [],
                "config_sha256": _config_sha(),
                "manifest_sha256": _manifest_sha(),
                "budget_deadline": budget_info}
    error = None
    status = "FAILED"
    try:
        free = eff._require_free_headroom(
            cfg, minimum_mib=cfg["resource"]["preconstruction_free_min_mib"],
            label="quality engine construction")
        progress["preconstruction"] = free
        init_t0 = time.perf_counter()
        h = {"cfg": cfg, "batch": int(cfg["controls"]["batch_size"]),
             "llm": eff._make_engine(cfg, int(cfg["controls"]["batch_size"])),
             "identity_layers": [], "expected": {}, "sampler": sampler,
             "budget_deadline": deadline}
        h["model_runner"] = h["llm"].model_runner
        eff._install_recorder(h, [])
        progress["engine_init_seconds"] = round(time.perf_counter() - init_t0, 3)
        kv = int(h["llm"].config.num_kvcache_blocks)
        cal = len(h["llm"].scheduler.cal_blocks)
        # Worst-case group token demand in 256-token pages, plus calc pages.
        largest4 = man["capacity"]["largest_four_prompts"]
        need = sum(math.ceil((p + cfg["controls"]["max_new_tokens"]) / 256)
                   for p in largest4)
        progress["kv_cache"] = {"blocks": kv, "cal_blocks": cal,
                                "free_data_blocks": kv - cal,
                                "needed_data_blocks_worst_group": need}
        if kv - cal < need:
            raise RuntimeError(f"KV capacity {kv - cal} < worst group need {need}")
        memory_pre = _memory_pre_generation(torch)
        progress["finite_parameters"] = _params_finite(h["model_runner"].model)
        memory_pre = _finish_memory_scan(torch, memory_pre)
        progress["memory_pre_generation"] = memory_pre
        if progress["finite_parameters"]["nonfinite_elements"] != 0:
            raise RuntimeError("nonfinite model parameters before generation")
        audit = {"calls": 0, "nonfinite_calls": 0, "max_numel": 0}
        _install_finite_hook(h, audit)
        progress["logits_audit"] = audit
        for g in range(32):
            order = (cfg["groups"]["order_even_group_index"] if g % 2 == 0
                     else cfg["groups"]["order_odd_group_index"])
            gids = ids[g * 4:(g + 1) * 4]
            items = []
            for idx in gids:
                meta = by_index[idx]
                items.append({"index": idx,
                              "token_ids": meta["token_ids"],
                              "prompt_sha256": meta["prompt_sha256"],
                              "prompt_tokens": meta["prompt_tokens"]})
            for pos, cond in enumerate(order):
                path = _group_path(g, cond)
                done = _validate_completed_group(path, man, g, cond, cfg)
                if done is not None:
                    key = f"g{g:02d}:{cond}"
                    if key not in progress["completed_group_conditions"]:
                        progress["completed_group_conditions"].append(key)
                    continue
                if path.exists():
                    failed_dir = RUN_DIR / "failed"
                    failed_dir.mkdir(exist_ok=True)
                    dest = failed_dir / (path.name + f".preserved-{int(time.time())}")
                    shutil.move(str(path), str(dest))
                    progress.setdefault("preserved_failed_files", []).append(str(dest))
                _budget_guard(cfg, time.time() - t_start)
                commit = execution_commit
                res = _run_batch(h, cfg, items, cond,
                                 (cfg["controls"]["sparsity_ratio_s2"] if cond == "s2"
                                  else cfg["controls"]["sparsity_ratio_dense"]),
                                 g, pos, cfg["controls"]["max_new_tokens"], commit)
                _atomic_json(path, res)
                if res["status"] != "SUCCEEDED":
                    raise RuntimeError(
                        f"group {g} {cond} failed: {res['problems'][:4]}")
                key = f"g{g:02d}:{cond}"
                progress["completed_group_conditions"].append(key)
                progress["last_group"] = {"group_index": g, "condition": cond,
                                          "wall_seconds": res["wall_seconds"],
                                          "decode_seconds": res["decode_seconds"],
                                          "peak_allocated_mib": res["peak_allocated_mib"],
                                          "counters": res["counters"]}
                progress["elapsed_seconds"] = round(time.time() - t_start, 3)
                progress["sampler_live_breach"] = sampler.breach
                _atomic_json(GENERATION_PATH, progress)
                print(json.dumps({"progress": key,
                                  "completed": len(progress["completed_group_conditions"]),
                                  "elapsed_s": progress["elapsed_seconds"],
                                  "group_wall_s": res["wall_seconds"],
                                  "decode_s": res["decode_seconds"]}), flush=True)
        if audit["nonfinite_calls"] != 0 or audit["calls"] == 0:
            raise RuntimeError(f"nonfinite/missing logits audit: {audit}")
        if len(progress["completed_group_conditions"]) != 64:
            raise RuntimeError("not all 64 group-condition calls completed")
        status = "SUCCEEDED"
    except BudgetError as exc:
        status = "BLOCKED"
        error = repr(exc)
        progress["budget"] = _budget_state(cfg, time.time() - t_start)
    except Exception as exc:  # noqa: BLE001
        status = "FAILED"
        error = repr(exc)
        progress["traceback"] = traceback.format_exc()[-8000:]
    finally:
        progress["sampler"] = sampler.summary()
        if sampler.breach is not None and status == "SUCCEEDED":
            status = "FAILED"
            error = f"resource guard breach: {sampler.breach}"
        if h is not None:
            try:
                _release_finite_hook(h)
                eff._release_engine(h)
                torch.cuda.synchronize()
                progress["engine_allocated_after_release_mib"] = round(
                    float(torch.cuda.memory_allocated() / 2**20), 1)
            except Exception as exc:  # noqa: BLE001
                progress["engine_release_error"] = repr(exc)
        progress["status"] = status
        progress["error"] = error
        progress["finished"] = eff._now()
        progress["wall_seconds"] = round(time.time() - t_start, 3)
        progress["budget_final"] = _budget_state(cfg)
        progress["sampler_breach"] = sampler.breach
        progress["raw_group_files"] = sorted(p.name for p in RAW_DIR.glob("group-*.json"))
        _atomic_json(GENERATION_PATH, progress)
        _record_attempt("quality_generation", status, progress["wall_seconds"],
                        started, eff._now(), execution_commit=progress["execution_commit"],
                        error=error)
        _log_command("quality_generation", args.argv, status=status,
                     wall_seconds=progress["wall_seconds"])
    print(json.dumps({"status": status,
                      "completed": len(progress["completed_group_conditions"]),
                      "wall_seconds": progress["wall_seconds"],
                      "error": error}, indent=2))
    if status == "SUCCEEDED":
        return 0
    return 2 if status == "BLOCKED" else 1


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def _paired_bootstrap(delta, seed, reps):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(reps, len(delta)))
    boot = delta[idx].mean(axis=1) * 100.0
    return np.quantile(boot, [0.025, 0.975]).tolist()


def _exact_paired_bootstrap(delta):
    prob = np.array([(delta == -1).mean(), (delta == 0).mean(),
                     (delta == 1).mean()], dtype=float)
    pmf = np.array([1.0])
    for _ in range(len(delta)):
        pmf = np.convolve(pmf, prob)
    cdf = pmf.cumsum()
    support = np.arange(-len(delta), len(delta) + 1) / len(delta) * 100.0
    return [float(support[np.searchsorted(cdf, q)]) for q in (0.025, 0.975)]


def _mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return None
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * tail))


def cmd_summarize(args):
    cfg = load_config()
    man, by_index = _load_frozen(cfg)
    ids = man["dataset"]["ids"]
    group_rows = {}
    commits = set()
    group_walls = []
    for g in range(32):
        for cond in ("dense", "s2"):
            path = _group_path(g, cond)
            data = _validate_completed_group(path, man, g, cond, cfg)
            if data is None:
                raise RuntimeError(f"missing/incomplete group file: {path}")
            group_rows[(g, cond)] = data
            commits.add(data["execution_commit"])
            group_walls.append(data["wall_seconds"])
    rows = []
    for g in range(32):
        order = (cfg["groups"]["order_even_group_index"] if g % 2 == 0
                 else cfg["groups"]["order_odd_group_index"])
        for pos, cond in enumerate(order):
            data = group_rows[(g, cond)]
            for raw in data["items"]:
                index = int(raw["index"])
                meta = next(x for x in man["items"] if int(x["index"]) == index)
                rawp = by_index[index]
                gold_frac = qs.extract_gold(rawp["answer"])
                gold = qs.canonical_fraction(gold_frac)
                pred_info = qs.extract_prediction(raw["text"])
                pred_frac = pred_info["value"]
                pred = qs.canonical_fraction(pred_frac)
                correct = bool(pred_frac is not None and pred_frac == gold_frac)
                if raw["prompt_sha256"] != meta["prompt_sha256"]:
                    raise RuntimeError(f"prompt hash mismatch in output index {index}")
                rows.append({
                    "index": index,
                    "condition": cond,
                    "gold_numeric": gold,
                    "prediction_numeric": pred,
                    "correct": correct,
                    "new_tokens": int(raw["new_tokens"]),
                    "truncated": bool(raw["truncated"]),
                    "stop_reason": raw["finish_reason"],
                    "extraction_method": pred_info["method"],
                    "extraction_form": pred_info["form"],
                    "extraction_failure_reason": pred_info["failure_reason"],
                    "empty_output": bool(raw["empty_output"]),
                    "contains_mask_token": bool(raw["contains_mask_token"]),
                    "ids_in_vocab": bool(raw["ids_in_vocab"]),
                    "output_sha256": raw["output_sha256"],
                    "prompt_sha256": raw["prompt_sha256"],
                    "execution_commit": data["execution_commit"],
                    "group_index": g,
                    "group_condition_order": pos,
                })
    if len(rows) != 256:
        raise RuntimeError(f"expected 256 sanitized rows, got {len(rows)}")
    keys = [(r["index"], r["condition"]) for r in rows]
    if len(set(keys)) != 256:
        raise RuntimeError("duplicate (index,condition) row")
    if set(r["index"] for r in rows) != set(ids):
        raise RuntimeError("sanitized rows do not cover the frozen 128 IDs")
    dense = np.array([r["correct"] for r in rows if r["condition"] == "dense"], dtype=int)
    s2 = np.array([r["correct"] for r in rows if r["condition"] == "s2"], dtype=int)
    if len(dense) != 128 or len(s2) != 128:
        raise RuntimeError("condition row count is not 128")
    # rows are grouped in frozen ID order; align explicitly by ID.
    dmap = {r["index"]: r["correct"] for r in rows if r["condition"] == "dense"}
    smap = {r["index"]: r["correct"] for r in rows if r["condition"] == "s2"}
    dense = np.array([int(dmap[i]) for i in ids], dtype=int)
    s2 = np.array([int(smap[i]) for i in ids], dtype=int)
    delta = s2 - dense
    boot_ci = _paired_bootstrap(delta, cfg["bootstrap"]["seed"], cfg["bootstrap"]["reps"])
    exact_ci = _exact_paired_bootstrap(delta)
    disc = {"both_correct": int(((dense == 1) & (s2 == 1)).sum()),
            "dense_only": int(((dense == 1) & (s2 == 0)).sum()),
            "s2_only": int(((dense == 0) & (s2 == 1)).sum()),
            "neither": int(((dense == 0) & (s2 == 0)).sum())}
    conditions = {}
    for cond in ("dense", "s2"):
        rr = [r for r in rows if r["condition"] == cond]
        lens = np.array([r["new_tokens"] for r in rr], dtype=int)
        methods = Counter(r["extraction_method"] for r in rr)
        by_method = {}
        for m in methods:
            sel = [r for r in rr if r["extraction_method"] == m]
            by_method[m] = {"n": len(sel), "correct": int(sum(r["correct"] for r in sel))}
        conditions[cond] = {
            "correct": int(sum(r["correct"] for r in rr)),
            "n": 128,
            "em_percent": float(sum(r["correct"] for r in rr) / 128 * 100),
            "new_tokens_mean": float(lens.mean()),
            "new_tokens_median": float(np.median(lens)),
            "new_tokens_p95": float(np.quantile(lens, 0.95)),
            "new_tokens_max": int(lens.max()),
            "truncated": int(sum(r["truncated"] for r in rr)),
            "eos_stops": int(sum(r["stop_reason"] == "eos" for r in rr)),
            "other_stops": int(sum(r["stop_reason"] == "other" for r in rr)),
            "empty_outputs": int(sum(r["empty_output"] for r in rr)),
            "extraction_failures": int(sum(r["prediction_numeric"] is None for r in rr)),
            "extraction_method_counts": dict(methods),
            "by_method": by_method,
            "stop_reason_counts": dict(Counter(r["stop_reason"] for r in rr)),
            "contains_mask_token": int(sum(r["contains_mask_token"] for r in rr)),
            "out_of_vocab": int(sum(not r["ids_in_vocab"] for r in rr)),
        }
    validation = {
        "all_256_rows": len(rows) == 256,
        "unique_pairs": len(set(keys)) == 256,
        "all_128_ids": set(r["index"] for r in rows) == set(ids),
        "dense_n": conditions["dense"]["n"] == 128,
        "s2_n": conditions["s2"]["n"] == 128,
        "no_mask_tokens": all(not r["contains_mask_token"] for r in rows),
        "all_ids_in_vocab": all(r["ids_in_vocab"] for r in rows),
        "truncation_consistent": all((not r["truncated"]) or r["new_tokens"] == 4096
                                     for r in rows),
        "prompt_hashes_match": all(
            r["prompt_sha256"] == next(x for x in man["items"]
                                       if int(x["index"]) == r["index"])["prompt_sha256"]
            for r in rows),
    }
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "created_at": eff._now(),
        "git_head": _git_head(),
        "n": 128,
        "conditions": conditions,
        "paired_delta_pp": float(delta.mean() * 100),
        "paired_bootstrap_ci95_pp": boot_ci,
        "bootstrap": {"seed": cfg["bootstrap"]["seed"],
                      "reps": cfg["bootstrap"]["reps"],
                      "confidence": cfg["bootstrap"]["confidence"],
                      "method": "paired item resampling of {-1,0,+1} delta"},
        "exact_paired_bootstrap_ci95_pp": exact_ci,
        "exact_bootstrap_method": "128-fold convolution of the empirical paired delta distribution; inverse-CDF quantiles",
        "discordance": disc,
        "mcnemar_exact_p": _mcnemar_exact(disc["dense_only"], disc["s2_only"]),
        "more_than_5pp_loss": bool(delta.mean() * 100 < -5.0),
        "execution_commits": sorted(commits),
        "mixed_execution_commits": len(commits) > 1,
        "group_wall_seconds_sum": round(float(sum(group_walls)), 3),
        "validation": validation,
        "limitation": "fixed 128 heldout items, one seed, original undistilled TraDo-4B; this is exploratory and does not establish equivalence for losses <=5pp or reproduce a paper distilled-table result",
    }
    if not all(validation.values()):
        summary["status"] = "FAILED_VALIDATION"
    else:
        summary["status"] = "SUCCEEDED"
    per_item = {"schema_version": 1, "run_id": RUN_ID, "n": 256, "rows": rows}
    _atomic_json(EXP_DIR / "quality-per-item.json", per_item)
    _atomic_json(RUN_DIR / "quality-per-item.json", per_item)
    _atomic_json(EXP_DIR / "quality-summary.json", summary)
    _atomic_json(RUN_DIR / "quality-summary.json", summary)
    metrics = {
        "schema_version": 1, "run_id": RUN_ID,
        "primary_metric": "paired numeric exact-match difference (s2 minus dense), percentage points",
        "split": "GSM8K test fixed128 frozen IDs; no tuning",
        "seed": 42,
        "dense_em_percent": conditions["dense"]["em_percent"],
        "s2_em_percent": conditions["s2"]["em_percent"],
        "dense_correct": conditions["dense"]["correct"],
        "s2_correct": conditions["s2"]["correct"],
        "paired_delta_pp": summary["paired_delta_pp"],
        "paired_bootstrap_ci95_pp": boot_ci,
        "discordance": disc,
        "more_than_5pp_loss": summary["more_than_5pp_loss"],
        "n": 128,
    }
    _atomic_json(EXP_DIR / "metrics.json", metrics)
    _atomic_json(RUN_DIR / "metrics.json", metrics)
    # Raw-output manifest (hashes/paths only; raw text stays outside Git).
    raw_manifest = {"schema_version": 1, "run_id": RUN_ID,
                    "note": "raw generated text/token IDs stay outside Git under runs/<run_id>/raw",
                    "groups": []}
    for g in range(32):
        for cond in ("dense", "s2"):
            path = _group_path(g, cond)
            data = group_rows[(g, cond)]
            raw_manifest["groups"].append({
                "group_index": g, "condition": cond,
                "relpath": f"runs/{RUN_ID}/raw/{path.name}",
                "sha256": _sha256(path), "bytes": path.stat().st_size,
                "items": [int(x["index"]) for x in data["items"]],
                "output_sha256": [x["output_sha256"] for x in data["items"]],
            })
    _atomic_json(EXP_DIR / "raw-output-manifest.json", raw_manifest)
    _atomic_json(RUN_DIR / "raw-output-manifest.json", raw_manifest)
    _log_command("summarize", args.argv, status=summary["status"])
    print(json.dumps({"status": summary["status"],
                      "dense_em_percent": conditions["dense"]["em_percent"],
                      "s2_em_percent": conditions["s2"]["em_percent"],
                      "paired_delta_pp": summary["paired_delta_pp"],
                      "ci95": boot_ci, "discordance": disc}, indent=2))
    return 0 if summary["status"] == "SUCCEEDED" else 1


# ---------------------------------------------------------------------------
# ledger / commands / finalize
# ---------------------------------------------------------------------------
def cmd_ledger(args):
    cfg = load_config()
    attempts = _read_jsonl(ATTEMPTS_PATH)
    gen = _load_json(GENERATION_PATH) if GENERATION_PATH.exists() else {}
    total = sum(float(a.get("wall_seconds") or 0.0) for a in attempts)
    stage_h = total / 3600.0
    ledger = {
        "schema_version": 1, "run_id": RUN_ID, "created_at": eff._now(),
        "git_head": _git_head(),
        "attempts": attempts,
        "gpu_command_wall_seconds_sum": round(total, 3),
        "stage_gpu_hours": round(stage_h, 6),
        "campaign_before_gpu_hours": cfg["budget"]["campaign_before_gpu_hours"],
        "campaign_after_upper_gpu_hours": round(
            cfg["budget"]["campaign_before_gpu_hours"] + stage_h, 6),
        "campaign_limit_gpu_hours": cfg["budget"]["campaign_limit_gpu_hours"],
        "stage_limit_gpu_hours": cfg["budget"]["stage_gpu_hours"],
        "reserve_gpu_hours": cfg["budget"]["reserve_gpu_hours"],
        "generation": {
            "wall_seconds": gen.get("wall_seconds"),
            "status": gen.get("status"),
            "completed_group_conditions": len(gen.get("completed_group_conditions", [])),
            "kv_cache": gen.get("kv_cache"),
            "peak_allocated_mib": (gen.get("last_group") or {}).get("peak_allocated_mib"),
            "sampler": gen.get("sampler"),
            "sampler_breach": gen.get("sampler_breach"),
        },
        "note": "conservative sum of measured GPU-command walls including smoke and failed attempts; includes probes that initialized CUDA; CPU freeze/scorer/summary not charged",
    }
    ledger["within_stage_limit"] = stage_h <= cfg["budget"]["stage_gpu_hours"]
    ledger["within_campaign_limit_with_reserve"] = (
        cfg["budget"]["campaign_before_gpu_hours"] + stage_h
        <= cfg["budget"]["campaign_limit_gpu_hours"] - cfg["budget"]["reserve_gpu_hours"])
    _atomic_json(EXP_DIR / "resource-ledger.json", ledger)
    _atomic_json(RUN_DIR / "resource-ledger.json", ledger)
    _log_command("ledger", args.argv, status="SUCCEEDED")
    print(json.dumps({"stage_gpu_hours": ledger["stage_gpu_hours"],
                      "campaign_after_upper_gpu_hours": ledger["campaign_after_upper_gpu_hours"],
                      "within_stage_limit": ledger["within_stage_limit"]}, indent=2))
    return 0


def cmd_commands(args):
    entries = _read_jsonl(COMMAND_LOG_PATH)
    provenance = {
        "schema_version": 1, "run_id": RUN_ID, "created_at": eff._now(),
        "git_head": _git_head(),
        "task_sha256": _sha256(EXP_DIR / "task.json"),
        "config_sha256": _config_sha(),
        "manifest_sha256": _manifest_sha() if MANIFEST_PATH.exists() else None,
        "scorer_sha256": _scorer_sha(),
        "scorer_tests_sha256": _sha256(HERE / "quality_scorer_tests.py"),
        "environment": "isolated Python3.10 torch2.4.1+cu121 triton3.0 numpy1.26.4 transformers4.52.4 FlashInfer0.2.9 custom FlashAttention; no reinstall",
        "entries": entries,
    }
    _atomic_json(EXP_DIR / "commands-and-provenance.json", provenance)
    _atomic_json(RUN_DIR / "commands-and-provenance.json", provenance)
    _log_command("commands", args.argv, status="SUCCEEDED")
    print(json.dumps({"entries": len(entries), "git_head": provenance["git_head"]}, indent=2))
    return 0


def cmd_finalize(args):
    cfg = load_config()
    fields_path = EXP_DIR / "report-fields.json"
    if not fields_path.exists():
        raise RuntimeError("report-fields.json is required before finalize")
    fields = _load_json(fields_path)
    summary = _load_json(EXP_DIR / "quality-summary.json")
    ledger = _load_json(EXP_DIR / "resource-ledger.json")
    commits_doc = _load_json(EXP_DIR / "run-commits.json")
    head = _git_head()
    metrics = [
        {"config": "dense", "seed": 42, "split": "GSM8K test fixed128",
         "metric": "numeric exact match", "value": summary["conditions"]["dense"]["em_percent"],
         "unit": "percent"},
        {"config": "s2/tau2", "seed": 42, "split": "GSM8K test fixed128",
         "metric": "numeric exact match", "value": summary["conditions"]["s2"]["em_percent"],
         "unit": "percent"},
        {"config": "s2-dense", "seed": 42, "split": "GSM8K test fixed128",
         "metric": "paired numeric exact-match difference",
         "value": summary["paired_delta_pp"], "unit": "percentage points"},
    ]
    whitelist = [
        "quality-per-item.json", "quality-summary.json", "quality-config.json",
        "scorer-test-results.json", "input-manifest.json", "train-smoke.json",
        "resource-ledger.json", "commands-and-provenance.json",
        "raw-output-manifest.json", "metrics.json", "environment.json",
        "report-summary.md", "run-commits.json", "generation.json",
    ]
    artifacts = [name for name in whitelist if (RUN_DIR / name).is_file()]
    report = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": fields["status"],
        "summary": fields["summary"],
        "executed_commands": fields["executed_commands"],
        "metrics": metrics,
        "artifacts": artifacts,
        "deviations": fields.get("deviations", []),
        "failures": fields.get("failures", []),
        "resource_usage": {
            "wall_seconds": fields.get("wall_seconds"),
            "gpu_hours": ledger["stage_gpu_hours"],
            "gpu_command_wall_seconds": ledger["gpu_command_wall_seconds_sum"],
            "campaign_after_upper_gpu_hours": ledger["campaign_after_upper_gpu_hours"],
            "stage_gpu_hours_limit": cfg["budget"]["stage_gpu_hours"],
            "note": "wall_seconds is the measured run wall from report-fields; gpu_hours is the conservative sum of measured GPU-command walls including smoke/failures",
        },
        "next_recommendation": fields["next_recommendation"],
        "version_control": {
            "repository": commits_doc["repository"],
            "visibility": commits_doc["visibility"],
            "branch": commits_doc["branch"],
            "run_commits": commits_doc["run_commits"],
            "head_commit": head,
            "commits": commits_doc.get("commits", []),
            "push_status": "pending",
            "push_error": "relay mode: executor commits locally only; GPT pulls the private repository objects and verifies/pushes",
        },
    }
    # Atomic write of the final report with a temp file.
    tmp = RUN_DIR / "report.json.tmp"
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, RUN_DIR / "report.json")
    print(json.dumps({"report": str(RUN_DIR / "report.json"),
                      "status": report["status"],
                      "head_commit": head,
                      "gpu_hours": report["resource_usage"]["gpu_hours"]}, indent=2))
    return 0


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scorer-tests")
    p.set_defaults(fn=cmd_scorer_tests)

    p = sub.add_parser("freeze")
    p.set_defaults(fn=cmd_freeze)

    p = sub.add_parser("probe")
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("smoke")
    p.add_argument("--manifest-sha", required=True)
    p.set_defaults(fn=cmd_smoke)

    p = sub.add_parser("generate")
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("summarize")
    p.set_defaults(fn=cmd_summarize)

    p = sub.add_parser("ledger")
    p.set_defaults(fn=cmd_ledger)

    p = sub.add_parser("commands")
    p.set_defaults(fn=cmd_commands)

    p = sub.add_parser("finalize")
    p.set_defaults(fn=cmd_finalize)

    args = ap.parse_args()
    args.argv = sys.argv
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
