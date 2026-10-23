#!/usr/bin/env python3
"""Efficiency stage CLI for flashblock-20261023-efficiency.

Frozen workload: 6 settings (B1/B4 x per-sequence 1024/4096/8192 tokens),
fixed 256 new tokens per request, dense (ratio 0) vs official s2 (tau=2),
5 paired interleaved rounds per setting, seed 42.  Sequential prefill for both
conditions.  Primary metric is mean paired dense_decode_seconds /
s2_decode_seconds at B4L8192.

Subcommands
  freeze     build and hash deterministic token manifests from GSM8K train text
  probe      small environment/headroom snapshot
  preflight  cached-slot identity + partial-completion + tiny timing sanity
  matrix     the frozen 60 timed rows (plus 12 warmup rows)
  profile    one separate B4L8192 profile per condition
  summarize  paired statistics and small exports
  finalize   write report.json with version-control block
"""
import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLASH = ROOT / "flash-attention"
DLLM = ROOT / "dllm"
RUN_ID = "flashblock-20261023-efficiency"
CONFIG_PATH = ROOT / "experiments" / RUN_ID / "efficiency-config.json"
RUN_DIR = ROOT / "runs" / RUN_ID
EXP_DIR = ROOT / "experiments" / RUN_ID

torch = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _init_torch():
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch
    return torch


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


def _setup_paths():
    for p in (str(FLASH), str(DLLM), str(DLLM / "sample")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _seed_all(seed):
    _init_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return {"seed": seed, "torch_initial_seed": int(torch.initial_seed()),
            "torch_cuda_initial_seed": int(torch.cuda.initial_seed())}


def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ids_hash(ids):
    return _sha256_bytes(np.asarray(ids, dtype=np.int64).tobytes())


def _atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _git_head():
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def _git_status():
    return subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip()


def _load_gsm8k_rows(parquet, indices):
    import pyarrow.parquet as pq
    rows = pq.read_table(parquet).to_pylist()
    return [dict(rows[i], _index=int(i)) for i in indices]


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _mean(v):
    return float(sum(v) / len(v)) if v else None


def _median(v):
    return float(np.median(v)) if v else None


# ---------------------------------------------------------------------------
# NVML lightweight sampler
# ---------------------------------------------------------------------------
class NvmlSampler:
    def __init__(self, gpu_uuid, interval=0.5, min_free_mib=3072, max_alloc_mib=16384):
        self.gpu_uuid = gpu_uuid
        self.interval = interval
        self.min_free_mib = float(min_free_mib)
        self.max_alloc_mib = float(max_alloc_mib)
        self.stop = None
        self.thread = None
        self.samples = []
        self.breach = None
        self._nvml = None
        self._handle = None

    def _init_nvml(self):
        import pynvml
        pynvml.nvmlInit()
        self._nvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByUUID(self.gpu_uuid)

    def query(self):
        m = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
        u = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
        return {"mem_total_mib": m.total / 2**20, "mem_used_mib": m.used / 2**20,
                "mem_free_mib": m.free / 2**20, "util_gpu_pct": int(u.gpu)}

    def _loop(self):
        import threading
        while not self.stop.is_set():
            try:
                q = self.query()
                q["t"] = time.time()
                self.samples.append(q)
                if q["mem_free_mib"] < self.min_free_mib and self.breach is None:
                    self.breach = {"kind": "physical_free_mib", "value": q["mem_free_mib"],
                                   "limit": self.min_free_mib, "at": _now()}
                if torch is not None:
                    alloc = torch.cuda.memory_allocated() / 2**20
                    if alloc > self.max_alloc_mib and self.breach is None:
                        self.breach = {"kind": "torch_allocated_mib", "value": alloc,
                                       "limit": self.max_alloc_mib, "at": _now()}
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        import threading
        self._init_nvml()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def summary(self):
        if self.stop is not None:
            self.stop.set()
            if self.thread is not None:
                self.thread.join(timeout=3)
        s = self.samples
        end = None
        try:
            end = self.query()
        except Exception:
            pass
        return {
            "samples": len(s),
            "mem_free_min_mib": round(min(x["mem_free_mib"] for x in s), 1) if s else None,
            "mem_used_max_mib": round(max(x["mem_used_mib"] for x in s), 1) if s else None,
            "util_gpu_mean_pct": round(float(np.mean([x["util_gpu_pct"] for x in s])), 2) if s else None,
            "util_gpu_max_pct": int(max(x["util_gpu_pct"] for x in s)) if s else None,
            "sample_first": s[0] if s else None,
            "sample_last": s[-1] if s else None,
            "end_query": end,
            "breach": self.breach,
            "limits": {"min_free_mib": self.min_free_mib, "max_allocated_mib": self.max_alloc_mib},
            "note": "physical NVML memory/util sampled identically both conditions; shared GPU",
        }


def _compute_apps(gpu_uuid):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu_uuid),
             "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            text=True, timeout=10).strip().splitlines()
    except Exception as e:
        return {"error": repr(e), "apps": []}
    apps = []
    for line in out:
        if not line.strip():
            continue
        pid, mem = [x.strip() for x in line.split(",")]
        apps.append({"pid": int(pid), "used_memory": mem})
    return {"count": len(apps), "apps": apps}


# ---------------------------------------------------------------------------
# input construction / manifest
# ---------------------------------------------------------------------------
def _build_inputs(cfg):
    _init_torch()
    from transformers import AutoTokenizer
    ds = cfg["dataset"]
    if _sha256(ds["train_parquet"]) != ds["train_sha256"]:
        raise RuntimeError("GSM8K train parquet sha256 mismatch")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["local_path"], use_fast=True, trust_remote_code=True)
    lengths = ds["lengths"]
    per = ds["inputs_per_length"]
    rows_per = ds["rows_per_input"]
    start = ds["input_index_start"]
    n_total = len(lengths) * per * rows_per
    rows = _load_gsm8k_rows(ds["train_parquet"], list(range(start, start + n_total)))
    raw_dir = RUN_DIR / "inputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "run_id": RUN_ID,
        "dataset": {k: ds[k] for k in ("dataset", "config", "revision", "train_parquet",
                                       "train_sha256", "split")},
        "recipe": ds["field_recipe"],
        "tokenizer": {"local_path": cfg["model"]["local_path"],
                      "class": type(tokenizer).__name__,
                      "vocab_size": int(tokenizer.vocab_size),
                      "mask_token_id": tokenizer.mask_token_id,
                      "pad_token_id": tokenizer.pad_token_id,
                      "eos_token_id": tokenizer.eos_token_id},
        "lengths": lengths, "inputs_per_length": per,
        "note": "raw token ids stay outside Git under runs/<run_id>/inputs/; only hashes/recipe in Git",
        "groups": {},
    }
    for li, L in enumerate(lengths):
        group = {"length": L, "inputs": []}
        prefixes = []
        for j in range(per):
            ridx = start + (li * per + j) * rows_per
            seg = rows[(li * per + j) * rows_per:(li * per + j + 1) * rows_per]
            text = "\n\n".join(r["question"] + "\n" + r["answer"] for r in seg)
            ids = tokenizer.encode(text)
            if len(ids) < L:
                raise RuntimeError(f"not enough tokens for L={L} input {j}: {len(ids)}")
            ids = [int(x) for x in ids[:L]]
            if len(ids) != L:
                raise RuntimeError("exact length failure")
            if tokenizer.mask_token_id in ids:
                raise RuntimeError("mask token leaked into workload input")
            first = ids[:256]
            prefixes.append(_ids_hash(first))
            raw = {"schema_version": 1, "length": L, "index": j,
                   "source_rows": [int(r["_index"]) for r in seg],
                   "row_index_start": int(ridx), "token_ids": ids}
            _atomic_json(raw_dir / f"ids-L{L}-j{j}.json", raw)
            full_hash = _ids_hash(ids)
            block_hashes = [_ids_hash(ids[k:k + 256]) for k in range(0, L, 256)]
            group["inputs"].append({
                "index": j, "length": L, "source_row_start": int(ridx),
                "source_row_end": int(ridx + rows_per - 1),
                "source_rows_sha256": _sha256_bytes(
                    np.asarray([r["_index"] for r in seg], dtype=np.int64).tobytes()),
                "raw_relpath": f"runs/{RUN_ID}/inputs/ids-L{L}-j{j}.json",
                "raw_sha256": _sha256(raw_dir / f"ids-L{L}-j{j}.json"),
                "token_ids_sha256": full_hash,
                "first256_sha256": prefixes[-1],
                "distinct_blocks": len(set(block_hashes)),
                "blocks": L // 256,
                "duplicate_blocks": (L // 256) - len(set(block_hashes)),
            })
        if len(set(prefixes)) != per:
            raise RuntimeError(f"shared first-256-token prefix at L={L}: {prefixes}")
        group["first256_distinct"] = True
        manifest["groups"][str(L)] = group
    _atomic_json(EXP_DIR / "input-manifest.json", manifest)
    return manifest


def _load_manifest(cfg):
    path = EXP_DIR / "input-manifest.json"
    with open(path, "r", encoding="utf-8") as f:
        man = json.load(f)
    tokens = {}
    for L, group in man["groups"].items():
        tokens[int(L)] = []
        for item in group["inputs"]:
            with open(ROOT / item["raw_relpath"], "r", encoding="utf-8") as f:
                raw = json.load(f)
            ids = raw["token_ids"]
            if len(ids) != int(L) or _ids_hash(ids) != item["token_ids_sha256"]:
                raise RuntimeError(f"manifest token mismatch L={L} j={item['index']}")
            tokens[int(L)].append(ids)
    return man, tokens


# ---------------------------------------------------------------------------
# engine / audited step driver
# ---------------------------------------------------------------------------
def _make_engine(cfg, batch):
    _init_torch()
    from jetengine_ext.llm import LLM
    c = cfg["controls"]
    return LLM(
        cfg["model"]["local_path"], enforce_eager=True, tensor_parallel_size=1,
        mask_token_id=151669, block_length=c["block_length"],
        batch_size=batch, gpu_memory_utilization=c["gpu_memory_utilization"],
        max_model_len=c["max_model_len"],
        max_num_batched_tokens=c["max_num_batched_tokens"],
        max_num_seqs=batch,
    )


def _reset_engine(h):
    sch = h["llm"].scheduler
    bm = sch.block_manager
    if sch.running:
        raise RuntimeError(f"reset with live requests: {len(sch.running)}")
    for b in bm.blocks:
        b.ref_count = 0
        b.hash = -1
        b.token_ids = []
    for cid in sch.cal_blocks:
        bm.blocks[cid].ref_count = 1
    cal = set(int(x) for x in sch.cal_blocks)
    bm.used_block_ids = set(cal)
    bm.free_block_ids = deque(i for i in range(len(bm.blocks)) if i not in cal)
    bm.hash_to_block_id = {}
    sch.token_ids_backend = {}
    sch.token_num_backend = {}
    sch.block_table_backend = {}
    sch.first_unmask_steps_backend = {}
    sch.current_block_tensor = None
    sch.count = 0
    sch.last_post_profile = {}
    sch.reuse_dump_count = 0
    sch.consistent_sampling_params = True
    h["model_runner"].attn_output_past.zero_()
    h["model_runner"].logsumexp.zero_()
    h["model_runner"].logsumexp_all.zero_()
    h["expected"] = {}
    torch.cuda.synchronize()


def _sid_rec(h, sid):
    return h.setdefault("per_seq_identity", {}).setdefault(
        int(sid), {"stores": 0, "reuses": 0, "reorder_checks": 0})


def _install_recorder(h, identity_layers):
    sch = h["llm"].scheduler
    orig = sch.schedule
    mr = h["model_runner"]

    def wrapped():
        running = list(sch.running)
        pre = []
        for i, s in enumerate(running):
            sp = s.sparsity_params or {}
            pre.append({
                "seq_id": int(s.seq_id), "row": i, "status": s.status.name,
                "num_tokens": int(s.num_tokens),
                "current_denoising_step": int(s.current_denoising_step),
                "dirty_count": int(sp.get("dirty_token_count", len(sp.get("dirty_tokens", [])))),
            })
        pre_rows, pre_lse = {}, {}
        if identity_layers:
            n = len(running)
            for layer in identity_layers:
                pre_rows[layer] = mr.attn_output_past[layer, :n].clone()
                pre_lse[layer] = mr.logsumexp[layer, :n].clone()
        seqs, rt = orig()
        rec = {"run_type": rt.name if rt is not None else None, "pre": pre,
               "issued": [int(s.seq_id) for s in seqs] if seqs else [],
               "no_update": [], "need_update": [], "need_store": [], "seq_ids": []}
        identity_on = bool(h.get("identity_enabled")) and bool(identity_layers)
        if seqs and rt is not None and rt.name == "DENOISE" and seqs[0].sparsity_params is not None:
            sp = seqs[0].sparsity_params
            rec["no_update"] = [int(x) for x in sp.get("no_need_update_kvcache_idx", [])]
            rec["need_update"] = [int(x) for x in sp.get("need_update_kvcache_idx", [])]
            rec["need_store"] = [int(x) for x in sp.get("need_store_history_idx", [])]
            rec["seq_ids"] = [int(x) for x in sp.get("seq_ids", [])]
            if identity_on and rec["no_update"]:
                if len(seqs) != len(running):
                    raise RuntimeError("identity preflight: batch order changed")
                for idx in rec["no_update"]:
                    seq = seqs[idx]
                    sid = int(seq.seq_id)
                    _sid_rec(h, sid)["reuses"] += 1
                    for layer in identity_layers:
                        key = (int(layer), sid)
                        if key not in h["expected"]:
                            raise RuntimeError(
                                f"identity failure: reuse before store seq={sid} layer={layer}")
                        want = h["expected"][key]
                        got_a = pre_rows[layer][idx]
                        got_l = pre_lse[layer][idx]
                        max_abs_a = float((got_a - want["a"]).abs().max().item())
                        max_abs_l = float((got_l - want["l"]).abs().max().item())
                        rel_a = float((got_a - want["a"]).norm().item() /
                                      max(1e-12, want["a"].norm().item()))
                        rel_l = float((got_l - want["l"]).norm().item() /
                                      max(1e-12, want["l"].norm().item()))
                        st = h["identity_stats"]
                        st["checks"] += 1
                        st["checks_a"] += 1
                        st["checks_l"] += 1
                        st["max_abs_a"] = max(st["max_abs_a"], max_abs_a)
                        st["max_abs_l"] = max(st["max_abs_l"], max_abs_l)
                        st["max_rel_l2_a"] = max(st["max_rel_l2_a"], rel_a)
                        st["max_rel_l2_l"] = max(st["max_rel_l2_l"], rel_l)
                        if max_abs_a != 0.0 or max_abs_l != 0.0:
                            raise RuntimeError(
                                f"identity failure: cached A/L mismatch seq={sid} layer={layer} "
                                f"max_abs_a={max_abs_a} max_abs_l={max_abs_l}")
        h["last_schedule"] = rec
        return seqs, rt

    sch.schedule = wrapped


def _step_h(h, ratio, identity=False):
    llm = h["llm"]
    sam = h.get("sampler")
    if sam is not None and sam.breach is not None:
        raise RuntimeError(f"resource guard breach: {sam.breach}")
    peak = float(torch.cuda.max_memory_allocated() / 2**20)
    if peak > h["cfg"]["controls"]["peak_allocated_limit_mib"]:
        raise RuntimeError(f"resource guard breach: torch allocated {peak:.1f} MiB")
    out = llm.step()
    rec = h["last_schedule"]
    cnt = h["counters"]
    if rec["run_type"] == "PREFILL":
        cnt["prefill_steps"] += 1
    elif rec["run_type"] == "DENOISE":
        cnt["decode_steps"] += 1
        cnt["decode_rows"] += len(rec["issued"])
        cnt["reuse_rows"] += len(rec["no_update"])
        cnt["update_rows"] += len(rec["need_update"])
        cnt["store_rows"] += len(rec["need_store"])
        cnt["saving_rows"] += sum(1 for p in rec["pre"] if p["status"] == "SAVING")
    finished = out[0]
    mr = h["model_runner"]
    if identity and h.get("identity_enabled", False):
        for idx in rec["need_store"]:
            sid = int(rec["seq_ids"][idx])
            _sid_rec(h, sid)["stores"] += 1
            for layer in h["identity_layers"]:
                h["expected"][(int(layer), sid)] = {
                    "a": mr.attn_output_past[layer, idx].clone(),
                    "l": mr.logsumexp[layer, idx].clone(),
                }
    if ratio > 0 and finished:
        from jetengine_ext.engine.llm_engine import move_to_end_torch
        idxs = [int(x[3]) for x in finished]
        perm = move_to_end_torch(h["batch"], idxs).to(mr.attn_output_past.device)
        mr.attn_output_past.copy_(mr.attn_output_past.index_select(1, perm))
        mr.logsumexp.copy_(mr.logsumexp.index_select(1, perm))
        if identity and h.get("identity_enabled", False):
            st = h["identity_stats"]
            st["reorders"] += 1
            survivors = list(llm.scheduler.running)
            for j, seq in enumerate(survivors):
                sid = int(seq.seq_id)
                _sid_rec(h, sid)["reorder_checks"] += 1
                for layer in h["identity_layers"]:
                    key = (int(layer), sid)
                    if key not in h["expected"]:
                        raise RuntimeError(f"reorder identity missing expected seq={sid}")
                    a = float((mr.attn_output_past[layer, j] - h["expected"][key]["a"]).abs().max().item())
                    l = float((mr.logsumexp[layer, j] - h["expected"][key]["l"]).abs().max().item())
                    st["reorder_checks"] += 1
                    st["max_reorder_abs_a"] = max(st["max_reorder_abs_a"], a)
                    st["max_reorder_abs_l"] = max(st["max_reorder_abs_l"], l)
                    if a != 0.0 or l != 0.0:
                        raise RuntimeError(
                            f"reorder identity failure seq={sid} layer={layer} "
                            f"max_abs_a={a} max_abs_l={l}")
    for item in finished:
        h["finished"][int(item[0])] = [int(x) for x in item[1]]
        h["finished_order"].append(int(item[0]))
    return out


def _make_seq(h, ids, sp, ratio):
    from jetengine_ext.engine.sequence import Sequence
    seq = Sequence(list(ids), h["llm"].config.mask_token_id, sp, ratio)
    seq.eos_token_id = h["llm"].tokenizer.eos_token_id
    return seq


def _run_once(h, prompts, max_tokens_list, ratio, identity=False, timeit=True,
              profile=False):
    _init_torch()
    import torch as T
    from jetengine_ext.sampling_params import SamplingParams
    llm = h["llm"]
    c = h["cfg"]["controls"]
    t_wall = time.perf_counter()
    h["identity_enabled"] = bool(identity)
    _reset_engine(h)
    h["finished"] = {}
    h["finished_order"] = []
    h["counters"] = Counter()
    h["per_seq_identity"] = {}
    h["inserted_seq_ids"] = []
    h["identity_stats"] = {"checks": 0, "checks_a": 0, "checks_l": 0,
                           "max_abs_a": 0.0, "max_abs_l": 0.0,
                           "max_rel_l2_a": 0.0, "max_rel_l2_l": 0.0,
                           "reorders": 0, "reorder_checks": 0,
                           "max_reorder_abs_a": 0.0, "max_reorder_abs_l": 0.0}
    seedinfo = _seed_all(c["seed"])
    sps = [SamplingParams(
        temperature=c["temperature"], max_tokens=int(mt), ignore_eos=bool(c["ignore_eos"]),
        block_length=c["block_length"], denoising_steps=c["denoising_steps"],
        dynamic_threshold=c["dynamic_threshold"], topk=c["top_k"], topp=c["top_p"],
        remasking_strategy=c["remasking_strategy"]) for mt in max_tokens_list]
    T.cuda.synchronize()
    T.cuda.reset_peak_memory_stats()
    load_probe = NvmlSampler(c["gpu_uuid"], interval=99)
    load_probe._init_nvml()
    load_start = load_probe.query()
    seqs = []
    initial_seq_ids = []
    pending = 0
    t0 = time.perf_counter()
    max_active = int(h["batch"])
    initial_batch = min(max_active, len(prompts))
    while pending < initial_batch:
        seq = _make_seq(h, prompts[pending], sps[pending], ratio)
        llm.scheduler.add(seq)
        seqs.append(seq)
        initial_seq_ids.append(int(seq.seq_id))
        _sid_rec(h, seq.seq_id)
        pending += 1
        guard = 0
        while seq.status.name == "WAITING":
            _step_h(h, ratio, identity=identity)
            guard += 1
            if guard > 4:
                raise RuntimeError("prefill did not complete in one/two steps")
    T.cuda.synchronize()
    prefill_seconds = time.perf_counter() - t0 if timeit else None

    prof = None
    if profile:
        from torch.profiler import profile as torch_profile, ProfilerActivity
        prof = torch_profile(activities=[ProfilerActivity.CUDA], record_shapes=False)
        prof.__enter__()
    T.cuda.synchronize()
    t_dec = time.perf_counter()
    while llm.scheduler.running or pending < len(prompts):
        while len(llm.scheduler.running) < max_active and pending < len(prompts):
            seq = _make_seq(h, prompts[pending], sps[pending], ratio)
            llm.scheduler.add(seq)
            seqs.append(seq)
            h["inserted_seq_ids"].append(int(seq.seq_id))
            _sid_rec(h, seq.seq_id)
            pending += 1
            guard = 0
            while seq.status.name == "WAITING":
                _step_h(h, ratio, identity=identity)
                guard += 1
                if guard > 4:
                    raise RuntimeError("mid-stream prefill did not complete")
        _step_h(h, ratio, identity=identity)
    T.cuda.synchronize()
    decode_seconds = time.perf_counter() - t_dec
    profile_rows, profile_totals = [], {}
    if prof is not None:
        prof.__exit__(None, None, None)
        ev = prof.key_averages()
        rows = []
        for e in ev:
            us = float(getattr(e, "self_cuda_time_total", 0.0) or 0.0)
            if us <= 0:
                continue
            rows.append({"key": e.key, "self_cuda_us": us, "count": int(e.count)})
        rows.sort(key=lambda x: -x["self_cuda_us"])
        total = sum(x["self_cuda_us"] for x in rows)
        pats = [p.lower() for p in h["cfg"]["profiling"]["attention_name_patterns"]]
        attn = sum(x["self_cuda_us"] for x in rows
                   if any(p in x["key"].lower() for p in pats))
        profile_rows = rows[:60]
        profile_totals = {"total_cuda_us": total, "attention_cuda_us": attn,
                          "other_cuda_us": total - attn,
                          "attention_fraction": (attn / total) if total else None,
                          "distinct_kernels": len(rows),
                          "classification_note": "attention keys contain one of "
                          + ",".join(pats) + "; approximate kernel-name classification"}
    load_end = load_probe.query()
    peak_alloc = float(T.cuda.max_memory_allocated() / 2**20)
    peak_res = float(T.cuda.max_memory_reserved() / 2**20)
    if h.get("sampler") is not None and h["sampler"].breach is not None:
        raise RuntimeError(f"resource guard breach at boundary: {h['sampler'].breach}")
    if peak_alloc > c["peak_allocated_limit_mib"]:
        raise RuntimeError(f"resource guard breach at boundary: allocated {peak_alloc}")
    if load_end["mem_free_mib"] < c["min_free_mib"]:
        raise RuntimeError(f"resource guard breach at boundary: physical free {load_end}")
    counts, hashes, problems = [], [], []
    all_out_ids = []
    for seq in seqs:
        toks = h["finished"].get(int(seq.seq_id))
        if toks is None:
            problems.append(f"seq {seq.seq_id} missing finished tokens")
            toks = []
        want = int(seq.max_tokens)
        counts.append(len(toks))
        all_out_ids.extend(toks)
        if len(toks) != want:
            problems.append(f"seq {seq.seq_id} tokens {len(toks)} != {want}")
        if any((x < 0 or x >= len(llm.tokenizer)) for x in toks):
            problems.append(f"seq {seq.seq_id} out-of-vocab id")
        if llm.config.mask_token_id in toks:
            problems.append(f"seq {seq.seq_id} mask id in output")
        hashes.append(_ids_hash(toks))
    for seq in seqs:
        _sid_rec(h, seq.seq_id)
    newseq_ok = [sid for sid in h["inserted_seq_ids"]
                 if h["per_seq_identity"][sid]["stores"] > 0
                 and h["per_seq_identity"][sid]["reuses"] > 0]
    survivors_ok = [sid for sid, v in h["per_seq_identity"].items()
                    if v["reorder_checks"] > 0]
    if identity and h["identity_stats"]["checks_a"] == 0:
        problems.append("identity preflight made no A reuse checks")
    if identity and h["identity_stats"]["checks_l"] == 0:
        problems.append("identity preflight made no L reuse checks")
    if identity and h["identity_stats"]["reorders"] == 0:
        problems.append("identity preflight made no completion reorder")
    return {
        "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
        "e2e_seconds": (prefill_seconds + decode_seconds) if timeit else None,
        "wall_seconds": round(time.perf_counter() - t_wall, 3),
        "output_token_counts": counts, "total_output_tokens": int(sum(counts)),
        "output_sha256_by_seq": hashes,
        "output_concat_sha256": _sha256_bytes(":".join(hashes).encode()),
        "max_output_token_id": max(all_out_ids) if all_out_ids else None,
        "tokenizer_len": len(llm.tokenizer),
        "tokenizer_vocab_size": int(llm.tokenizer.vocab_size),
        "problems": problems,
        "seed_before_run": seedinfo,
        "counters": {k: int(v) for k, v in sorted(h["counters"].items())},
        "per_seq_global_denoise_steps": [int(s.global_denoising_step) for s in seqs],
        "per_seq_completion_tokens": [int(s.num_completion_tokens) for s in seqs],
        "identity": h["identity_stats"] if identity else None,
        "initial_seq_ids": initial_seq_ids,
        "inserted_seq_ids": h["inserted_seq_ids"],
        "finished_order": h["finished_order"],
        "per_seq_identity": h["per_seq_identity"],
        "newseq_reuse_after_store": newseq_ok,
        "survivors_with_reorder": survivors_ok,
        "peak_allocated_mib": round(peak_alloc, 1),
        "peak_reserved_mib": round(peak_res, 1),
        "load_start": load_start, "load_end": load_end,
        "profile": ({"kernels": profile_rows, "totals": profile_totals} if profile else None),
    }


# ---------------------------------------------------------------------------
# freeze / probe
# ---------------------------------------------------------------------------
def cmd_freeze(args):
    cfg = load_config(args.config)
    _configure_env(cfg)
    _setup_paths()
    if _git_status():
        raise SystemExit("refusing freeze with dirty worktree")
    man = _build_inputs(cfg)
    print(json.dumps({"status": "SUCCEEDED", "manifest": str(EXP_DIR / "input-manifest.json"),
                      "groups": {k: len(v["inputs"]) for k, v in man["groups"].items()}}))
    return 0


def cmd_probe(args):
    _configure_env(load_config(args.config))
    _setup_paths()
    _init_torch()
    import importlib.metadata as md
    import shutil
    import platform
    pkgs = {}
    for name in ["torch", "triton", "numpy", "transformers", "tokenizers",
                 "huggingface-hub", "safetensors", "sentencepiece", "accelerate",
                 "pyarrow", "flashinfer-python", "flash-attn", "xxhash"]:
        try:
            pkgs[name] = md.version(name)
        except md.PackageNotFoundError:
            pkgs[name] = None
    s = NvmlSampler(load_config(args.config)["controls"]["gpu_uuid"], interval=99)
    s._init_nvml()
    q = s.query()
    payload = {
        "schema_version": 1, "run_id": RUN_ID, "git_head": _git_head(),
        "git_status": _git_status(), "branch": subprocess.check_output(
            ["git", "-C", str(ROOT), "branch", "--show-current"], text=True).strip(),
        "python": sys.version, "platform": platform.platform(), "packages": pkgs,
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda,
                  "cudnn": torch.backends.cudnn.version(),
                  "device_name": torch.cuda.get_device_name(0),
                  "capability": list(torch.cuda.get_device_capability(0))},
        "gpu": q, "gpu_uuid": load_config(args.config)["controls"]["gpu_uuid"],
        "compute_apps_before": _compute_apps(load_config(args.config)["controls"]["gpu_uuid"]),
        "cpu_count": os.cpu_count(),
        "disk_free_gib": round(shutil.disk_usage(str(ROOT)).free / 2**30, 3),
        "disk_total_gib": round(shutil.disk_usage("/mnt/data").total / 2**30, 3),
        "probed_at_utc": _now(),
    }
    _atomic_json(Path(args.out).resolve(), payload)
    print(json.dumps({"status": "SUCCEEDED", "out": args.out, "gpu": q,
                      "disk_free_gib": payload["disk_free_gib"]}))
    return 0


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
def cmd_preflight(args):
    cfg = load_config(args.config)
    _configure_env(cfg)
    _setup_paths()
    _init_torch()
    if _git_status():
        raise SystemExit("refusing GPU preflight with dirty worktree")
    commit = _git_head()
    man, tokens = _load_manifest(cfg)
    pf = cfg["preflight"]
    out = Path(args.out).resolve()
    res = {"schema_version": 1, "run_id": RUN_ID, "status": "FAILED",
           "execution_commit": commit, "argv": [sys.executable] + sys.argv,
           "started_at_utc": _now(), "batch": pf["batch"],
           "max_active": pf["max_active"]}
    t_wall = time.perf_counter()
    sampler = NvmlSampler(cfg["controls"]["gpu_uuid"],
                          min_free_mib=cfg["controls"]["min_free_mib"],
                          max_alloc_mib=cfg["controls"]["peak_allocated_limit_mib"])
    h = None
    try:
        sampler.start()
        h = {"cfg": cfg, "batch": pf["batch"], "llm": _make_engine(cfg, pf["batch"]),
             "identity_layers": pf["identity_layers"], "expected": {},
             "sampler": sampler}
        h["model_runner"] = h["llm"].model_runner
        _install_recorder(h, pf["identity_layers"])
        kv = int(h["llm"].config.num_kvcache_blocks)
        cal = len(h["llm"].scheduler.cal_blocks)
        need = pf["batch"] * math.ceil((cfg["controls"]["max_model_len"]) / 256)
        res["kv_cache"] = {"blocks": kv, "cal_blocks": cal, "block_size": 256,
                          "free_data_blocks": kv - cal, "needed_data_blocks": need}
        if kv - cal < need:
            raise RuntimeError(f"KV capacity {kv - cal} < required {need}")
        # ---- partial-completion / cache identity stream ----
        rows = _load_gsm8k_rows(cfg["dataset"]["train_parquet"], pf["indices"])
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["model"]["local_path"], use_fast=True,
                                            trust_remote_code=True)
        prompts = [tok.encode(r["question"] + "\n" + r["answer"])[:512] for r in rows]
        _reset_engine(h)
        h["batch"] = pf["max_active"]  # scheduler capacity is 4
        audit = {"calls": 0, "nonfinite_calls": 0, "sync_per_call": True}
        orig_run_model = h["model_runner"].run_model

        def audited_run_model(*a, **k):
            logits = orig_run_model(*a, **k)
            audit["calls"] += 1
            if not bool(torch.isfinite(logits).all().item()):
                audit["nonfinite_calls"] += 1
                raise RuntimeError(f"non-finite logits at audited call {audit['calls']}")
            return logits

        h["model_runner"].run_model = audited_run_model
        try:
            run = _run_once(h, prompts, pf["max_tokens"],
                            cfg["controls"]["sparsity_ratio_s2"], identity=True, timeit=False)
        finally:
            h["model_runner"].run_model = orig_run_model
        res["logits_audit"] = audit
        res["partial_completion"] = run
        if run["problems"]:
            raise RuntimeError(f"partial-completion problems: {run['problems'][:5]}")
        if run["total_output_tokens"] != sum(pf["max_tokens"]):
            raise RuntimeError("partial-completion token count mismatch")
        expected_stores = sum(1 for x in run["per_seq_completion_tokens"] if x > 0)
        res["identity_requirements"] = {
            "A_reuse_checks_gt_0": run["identity"]["checks_a"] > 0,
            "L_reuse_checks_gt_0": run["identity"]["checks_l"] > 0,
            "reorder_checks_gt_0": run["identity"]["reorder_checks"] > 0,
            "new_request_inserted": len(run["inserted_seq_ids"]) > 0,
            "new_request_reuse_after_store": len(run["newseq_reuse_after_store"]) > 0,
            "all_outputs_exact": run["output_token_counts"] == pf["max_tokens"],
            "zero_A_identity_mismatch": run["identity"]["max_abs_a"] == 0.0,
            "zero_L_identity_mismatch": run["identity"]["max_abs_l"] == 0.0,
            "zero_reorder_A_mismatch": run["identity"]["max_reorder_abs_a"] == 0.0,
            "zero_reorder_L_mismatch": run["identity"]["max_reorder_abs_l"] == 0.0,
        }
        if not all([res["identity_requirements"]["A_reuse_checks_gt_0"],
                    res["identity_requirements"]["L_reuse_checks_gt_0"],
                    res["identity_requirements"]["new_request_reuse_after_store"],
                    res["identity_requirements"]["zero_A_identity_mismatch"],
                    res["identity_requirements"]["zero_L_identity_mismatch"]]):
            raise RuntimeError(f"identity requirements failed: {res['identity_requirements']}")
        # ---- tiny fixed-work timing sanity, dense then s2 ----
        tiny = {"batch": pf["tiny_batch"], "length": pf["tiny_length"],
                "max_tokens": pf["tiny_max_tokens"], "runs": []}
        for cond in cfg["conditions"]:
            ratio = float(cond["ratio"])
            # same engine, just reset; batch size matches 4
            rr = _run_once(h, tokens[pf["tiny_length"]][:pf["tiny_batch"]],
                           [pf["tiny_max_tokens"]] * pf["tiny_batch"], ratio,
                           identity=False, timeit=True)
            tiny["runs"].append({"condition": cond["name"], "ratio": ratio,
                                 "decode_seconds": rr["decode_seconds"],
                                 "prefill_seconds": rr["prefill_seconds"],
                                 "total_output_tokens": rr["total_output_tokens"],
                                 "problems": rr["problems"]})
            if rr["problems"]:
                raise RuntimeError(f"tiny timing problems {cond['name']}: {rr['problems'][:3]}")
            if not (rr["decode_seconds"] and rr["decode_seconds"] > 0
                    and rr["prefill_seconds"] is not None and rr["prefill_seconds"] > 0):
                raise RuntimeError("nonpositive tiny timing")
        res["tiny_timing"] = tiny
        res["finite_parameters"] = {"checked": False,
                                    "note": "weights accepted by parent mechanism stage; every "
                                            "run_model logit tensor was checked with torch.isfinite "
                                            "in this untimed preflight (per-call sync disclosed)"}
        res["status"] = "SUCCEEDED"
    except Exception as e:
        import traceback
        res["error"] = repr(e)
        res["traceback"] = traceback.format_exc()[-5000:]
        res["status"] = "FAILED"
    finally:
        try:
            if h is not None:
                h["llm"].exit()
        except Exception as e:
            res["engine_exit_error"] = repr(e)
        res["resources"] = sampler.summary()
        res["wall_seconds"] = round(time.perf_counter() - t_wall, 3)
        res["gpu_command_wall_seconds"] = res["wall_seconds"]
        res["headroom"] = {"compute_apps_after": _compute_apps(cfg["controls"]["gpu_uuid"])}
        res["ended_at_utc"] = _now()
        _atomic_json(out, res)
        if res["status"] == "SUCCEEDED":
            compact = {k: v for k, v in res.items() if k not in ("partial_completion",)}
            compact["partial_completion"] = {
                k: v for k, v in res["partial_completion"].items()
                if k not in ("load_start", "load_end", "output_sha256_by_seq",
                             "seed_before_run", "profile")
            }
            _atomic_json(EXP_DIR / "preflight.json", compact)
    print(json.dumps({"status": res["status"], "out": str(out), "error": res.get("error")}))
    return 0 if res["status"] == "SUCCEEDED" else 1


# ---------------------------------------------------------------------------
# matrix
# ---------------------------------------------------------------------------
def _prior_gpu_wall(include_current_matrix=False):
    total = 0.0
    details = {}
    for p in sorted(RUN_DIR.glob("preflight*.json")):
        try:
            d = _load(p)
            w = float(d.get("gpu_command_wall_seconds") or d.get("wall_seconds") or 0.0)
            details[p.name] = w
            total += w
        except Exception:
            pass
    for name in ("matrix.json", "profile.json"):
        p = RUN_DIR / name
        if p.exists() and (include_current_matrix or name != "matrix.json"):
            try:
                d = _load(p)
                w = float(d.get("gpu_command_wall_seconds") or 0.0)
                details[name] = w
                total += w
            except Exception:
                pass
    return total, details


def cmd_matrix(args):
    cfg = load_config(args.config)
    _configure_env(cfg)
    _setup_paths()
    _init_torch()
    if _git_status():
        raise SystemExit("refusing GPU matrix with dirty worktree")
    commit = _git_head()
    man, tokens = _load_manifest(cfg)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    prior_wall, prior_detail = _prior_gpu_wall()
    rows_path = RUN_DIR / "timed-rows.json"
    all_rows = []
    failures = []
    current_h = None
    sampler = NvmlSampler(cfg["controls"]["gpu_uuid"],
                          min_free_mib=cfg["controls"]["min_free_mib"],
                          max_alloc_mib=cfg["controls"]["peak_allocated_limit_mib"])
    meta = {"schema_version": 1, "run_id": RUN_ID, "execution_commit": commit,
            "argv": [sys.executable] + sys.argv, "started_at_utc": _now(),
            "input_manifest_sha256": _sha256(EXP_DIR / "input-manifest.json"),
            "config_sha256": _sha256(CONFIG_PATH), "warmups": [], "rows": [],
            "failures": [], "status": "FAILED",
            "prior_gpu_wall_seconds": round(prior_wall, 3),
            "prior_gpu_wall_detail": prior_detail,
            "gpu_command_wall_limit_seconds": cfg["budget"]["gpu_command_wall_limit_seconds"]}
    try:
        sampler.start()
        for batch in [1, 4]:
            settings = [s for s in cfg["settings"] if int(s["batch"]) == batch]
            current_h = {"cfg": cfg, "batch": batch, "llm": _make_engine(cfg, batch),
                         "identity_layers": [], "expected": {}, "sampler": sampler}
            current_h["model_runner"] = current_h["llm"].model_runner
            _install_recorder(current_h, [])
            kv = int(current_h["llm"].config.num_kvcache_blocks)
            cal = len(current_h["llm"].scheduler.cal_blocks)
            need = batch * math.ceil(cfg["controls"]["max_model_len"] / 256)
            meta[f"kv_cache_B{batch}"] = {"blocks": kv, "cal_blocks": cal,
                                          "free_data_blocks": kv - cal,
                                          "needed_data_blocks": need}
            if kv - cal < need:
                raise RuntimeError(f"B{batch} KV capacity {kv-cal} < {need}")
            for setting in settings:
                L, B = int(setting["length"]), int(setting["batch"])
                prompts = tokens[L][:B]
                # warmup both conditions at this exact shape (untimed)
                for cond in cfg["conditions"]:
                    rr = _run_once(current_h, prompts,
                                   [cfg["controls"]["max_new_tokens"]] * B,
                                   float(cond["ratio"]), identity=False, timeit=True)
                    w = {"stage": "warmup", "setting": setting["name"],
                         "condition": cond["name"],
                         "decode_seconds": rr["decode_seconds"],
                         "prefill_seconds": rr["prefill_seconds"],
                         "wall_seconds": rr["wall_seconds"],
                         "total_output_tokens": rr["total_output_tokens"],
                         "peak_allocated_mib": rr["peak_allocated_mib"],
                         "problems": rr["problems"], "execution_commit": commit,
                         "ended_at_utc": _now()}
                    meta["warmups"].append(w)
                    if rr["problems"]:
                        raise RuntimeError(f"warmup {setting['name']} {cond['name']}: "
                                           f"{rr['problems'][:3]}")
                for ri, order in enumerate(cfg["pair_rounds"]):
                    for cname in order:
                        cond = next(c for c in cfg["conditions"] if c["name"] == cname)
                        elapsed = time.perf_counter() - t_start
                        cumulative = prior_wall + elapsed
                        if cumulative > cfg["budget"]["gpu_command_wall_limit_seconds"]:
                            raise RuntimeError(
                                f"cumulative GPU-command wall budget exhausted: "
                                f"{cumulative:.1f}s > "
                                f"{cfg['budget']['gpu_command_wall_limit_seconds']}s")
                        rr = _run_once(current_h, prompts,
                                       [cfg["controls"]["max_new_tokens"]] * B,
                                       float(cond["ratio"]), identity=False, timeit=True)
                        row = {"schema_version": 1, "run_id": RUN_ID,
                               "setting": setting["name"], "batch": B, "length": L,
                               "round": ri + 1, "order_index": order.index(cname),
                               "condition": cname, "ratio": float(cond["ratio"]),
                               "seed": cfg["controls"]["seed"],
                               "max_new_tokens": cfg["controls"]["max_new_tokens"],
                               "ignore_eos": True,
                               "prefill_seconds": rr["prefill_seconds"],
                               "decode_seconds": rr["decode_seconds"],
                               "e2e_seconds": rr["e2e_seconds"],
                               "wall_seconds": rr["wall_seconds"],
                               "decode_tps": (rr["total_output_tokens"] / rr["decode_seconds"]
                                              if rr["decode_seconds"] else None),
                               "total_output_tokens": rr["total_output_tokens"],
                               "output_token_counts": rr["output_token_counts"],
                               "output_concat_sha256": rr["output_concat_sha256"],
                               "counters": rr["counters"],
                               "per_seq_global_denoise_steps": rr["per_seq_global_denoise_steps"],
                               "per_seq_completion_tokens": rr["per_seq_completion_tokens"],
                               "peak_allocated_mib": rr["peak_allocated_mib"],
                               "peak_reserved_mib": rr["peak_reserved_mib"],
                               "load_start": rr["load_start"], "load_end": rr["load_end"],
                               "problems": rr["problems"],
                               "execution_commit": commit, "ended_at_utc": _now()}
                        if row["decode_seconds"] is None or row["prefill_seconds"] is None:
                            raise RuntimeError("missing timing")
                        if not math.isfinite(row["decode_seconds"]) or row["decode_seconds"] <= 0:
                            raise RuntimeError(f"nonpositive/nonfinite decode time {row}")
                        if rr["total_output_tokens"] != B * cfg["controls"]["max_new_tokens"]:
                            raise RuntimeError(
                                f"output count {rr['total_output_tokens']} != "
                                f"{B * cfg['controls']['max_new_tokens']}")
                        if row["problems"]:
                            raise RuntimeError(f"row problems: {row['problems'][:3]}")
                        all_rows.append(row)
                        meta["rows"] = all_rows
                        _atomic_json(rows_path, {"schema_version": 1, "run_id": RUN_ID,
                                                 "execution_commit": commit,
                                                 "rows": all_rows})
                        print(json.dumps({"setting": setting["name"], "round": ri + 1,
                                          "condition": cname,
                                          "decode_s": round(row["decode_seconds"], 3),
                                          "prefill_s": round(row["prefill_seconds"], 3),
                                          "tokens": row["total_output_tokens"]}), flush=True)
                _atomic_json(RUN_DIR / "matrix-progress.json", {
                    "status": "RUNNING", "execution_commit": commit,
                    "completed_rows": len(all_rows), "warmups": len(meta["warmups"]),
                    "gpu_wall_seconds": round(time.perf_counter() - t_start, 1)})
            # release this engine completely before constructing the next batch engine
            try:
                current_h["llm"].exit()
            finally:
                current_h["model_runner"] = None
                current_h["llm"] = None
                current_h = None
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        meta["status"] = "SUCCEEDED"
    except Exception as e:
        import traceback
        failures.append({"error": repr(e), "traceback": traceback.format_exc()[-5000:],
                         "at_utc": _now(), "completed_rows": len(all_rows)})
        meta["failures"] = failures
        meta["status"] = "FAILED"
    finally:
        if current_h is not None:
            try:
                current_h["llm"].exit()
            except Exception as e:
                failures.append({"engine_exit": "error", "error": repr(e)})
        meta["resources"] = sampler.summary()
        meta["gpu_command_wall_seconds"] = round(time.perf_counter() - t_start, 3)
        meta["cumulative_gpu_wall_seconds"] = round(
            prior_wall + meta["gpu_command_wall_seconds"], 3)
        meta["ended_at_utc"] = _now()
        meta["rows"] = all_rows
        _atomic_json(RUN_DIR / "matrix-progress.json", meta)
        _atomic_json(rows_path, {"schema_version": 1, "run_id": RUN_ID,
                                 "execution_commit": commit, "status": meta["status"],
                                 "rows": all_rows})
        _atomic_json(RUN_DIR / "matrix.json", meta)
    print(json.dumps({"status": meta["status"], "rows": len(all_rows),
                      "gpu_wall_seconds": meta["gpu_command_wall_seconds"],
                      "cumulative_gpu_wall_seconds": meta["cumulative_gpu_wall_seconds"],
                      "failures": failures[:1]}))
    return 0 if meta["status"] == "SUCCEEDED" else 1


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------
def cmd_profile(args):
    cfg = load_config(args.config)
    _configure_env(cfg)
    _setup_paths()
    _init_torch()
    if _git_status():
        raise SystemExit("refusing GPU profile with dirty worktree")
    commit = _git_head()
    man, tokens = _load_manifest(cfg)
    out = Path(args.out).resolve()
    t_wall = time.perf_counter()
    prior_wall, prior_detail = _prior_gpu_wall()
    res = {"schema_version": 1, "run_id": RUN_ID, "status": "FAILED",
           "execution_commit": commit, "argv": [sys.executable] + sys.argv,
           "started_at_utc": _now(), "runs": [], "warmups": [],
           "prior_gpu_wall_seconds": round(prior_wall, 3),
           "prior_gpu_wall_detail": prior_detail}
    sampler = NvmlSampler(cfg["controls"]["gpu_uuid"],
                          min_free_mib=cfg["controls"]["min_free_mib"],
                          max_alloc_mib=cfg["controls"]["peak_allocated_limit_mib"])
    h = None
    try:
        sampler.start()
        h = {"cfg": cfg, "batch": 4, "llm": _make_engine(cfg, 4),
             "identity_layers": [], "expected": {}, "sampler": sampler}
        h["model_runner"] = h["llm"].model_runner
        _install_recorder(h, [])
        prompts = tokens[8192][:4]
        for cond in cfg["conditions"]:
            rr = _run_once(h, prompts, [256] * 4, float(cond["ratio"]),
                           identity=False, timeit=True, profile=False)
            res["warmups"].append({"condition": cond["name"],
                                   "decode_seconds": rr["decode_seconds"],
                                   "prefill_seconds": rr["prefill_seconds"],
                                   "wall_seconds": rr["wall_seconds"],
                                   "problems": rr["problems"]})
            if rr["problems"]:
                raise RuntimeError(f"profile warmup {cond['name']}: {rr['problems'][:3]}")
        for cond in cfg["conditions"]:
            cumulative = prior_wall + (time.perf_counter() - t_wall)
            if cumulative > cfg["budget"]["gpu_command_wall_limit_seconds"]:
                raise RuntimeError(f"cumulative GPU-command wall budget exhausted before profile: "
                                   f"{cumulative:.1f}s")
            rr = _run_once(h, prompts, [256] * 4, float(cond["ratio"]),
                           identity=False, timeit=True, profile=True)
            item = {"condition": cond["name"], "ratio": float(cond["ratio"]),
                    "decode_seconds": rr["decode_seconds"],
                    "prefill_seconds": rr["prefill_seconds"],
                    "e2e_seconds": rr["e2e_seconds"],
                    "wall_seconds": rr["wall_seconds"],
                    "total_output_tokens": rr["total_output_tokens"],
                    "counters": rr["counters"],
                    "per_seq_global_denoise_steps": rr["per_seq_global_denoise_steps"],
                    "peak_allocated_mib": rr["peak_allocated_mib"],
                    "problems": rr["problems"], "profile": rr["profile"]}
            res["runs"].append(item)
            if rr["problems"]:
                raise RuntimeError(f"profile {cond['name']}: {rr['problems'][:3]}")
            if len(rr["output_token_counts"]) != 4 or any(
                    c != 256 for c in rr["output_token_counts"]):
                raise RuntimeError("profile output count")
        res["status"] = "SUCCEEDED"
    except Exception as e:
        import traceback
        res["error"] = repr(e)
        res["traceback"] = traceback.format_exc()[-5000:]
    finally:
        try:
            if h is not None:
                h["llm"].exit()
        except Exception as e:
            res["engine_exit_error"] = repr(e)
        res["resources"] = sampler.summary()
        res["gpu_command_wall_seconds"] = round(time.perf_counter() - t_wall, 3)
        res["cumulative_gpu_wall_seconds"] = round(
            prior_wall + res["gpu_command_wall_seconds"], 3)
        res["ended_at_utc"] = _now()
        _atomic_json(out, res)
        if res["status"] == "SUCCEEDED":
            small = {"schema_version": 1, "run_id": RUN_ID,
                     "execution_commit": commit,
                     "note": "separate B4L8192 profiles after warmup; latencies excluded "
                             "from 60 primary rows",
                     "runs": [{"condition": r["condition"],
                               "decode_seconds": r["decode_seconds"],
                               "prefill_seconds": r["prefill_seconds"],
                               "wall_seconds": r["wall_seconds"],
                               "total_output_tokens": r["total_output_tokens"],
                               "counters": r["counters"],
                               "per_seq_global_denoise_steps":
                                   r["per_seq_global_denoise_steps"],
                               "peak_allocated_mib": r["peak_allocated_mib"],
                               "profile": r["profile"]} for r in res["runs"]]}
            _atomic_json(EXP_DIR / "separate-profile.json", small)
    print(json.dumps({"status": res["status"], "out": str(out),
                      "error": res.get("error")}))
    return 0 if res["status"] == "SUCCEEDED" else 1


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def _bootstrap_ci(values, reps, seed, confidence):
    rng = random.Random(seed)
    vals = [float(x) for x in values]
    n = len(vals)
    means = [sum(rng.choice(vals) for _ in range(n)) / n for _ in range(reps)]
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = means[int(math.floor(alpha * (len(means) - 1)))]
    hi = means[int(math.ceil((1 - alpha) * (len(means) - 1)))]
    return [lo, hi]


def cmd_summarize(args):
    cfg = load_config(args.config)
    with open(RUN_DIR / "timed-rows.json", "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload["rows"]
    if len(rows) != 60:
        raise SystemExit(f"expected 60 timed rows, got {len(rows)}")
    by = {}
    for r in rows:
        by.setdefault(r["setting"], {}).setdefault(r["round"], {})[r["condition"]] = r
    bcfg = cfg["bootstrap"]
    summary = {"schema_version": 1, "run_id": RUN_ID,
               "execution_commit": payload.get("execution_commit"),
               "primary_setting": cfg["primary_setting"],
               "decision_rule": cfg["decision_rule"],
               "bootstrap": bcfg, "settings": {}, "primary": {}, "all_rows": len(rows)}
    for s in cfg["settings"]:
        name = s["name"]
        pairs = []
        for rnd in range(1, 6):
            d = by[name][rnd]["dense"]
            u = by[name][rnd]["s2"]
            ratio = d["decode_seconds"] / u["decode_seconds"]
            pairs.append({"round": rnd, "dense_decode_seconds": d["decode_seconds"],
                          "s2_decode_seconds": u["decode_seconds"],
                          "paired_speedup": ratio,
                          "dense_tps": d["decode_tps"], "s2_tps": u["decode_tps"],
                          "dense_steps": sum(d["per_seq_global_denoise_steps"]),
                          "s2_steps": sum(u["per_seq_global_denoise_steps"]),
                          "dense_prefill_seconds": d["prefill_seconds"],
                          "s2_prefill_seconds": u["prefill_seconds"],
                          "dense_e2e_seconds": d["e2e_seconds"],
                          "s2_e2e_seconds": u["e2e_seconds"]})
        ratios = [p["paired_speedup"] for p in pairs]
        conds = {}
        for cname in ("dense", "s2"):
            sub = [by[name][rnd][cname] for rnd in range(1, 6)]
            dec = [x["decode_seconds"] for x in sub]
            pre = [x["prefill_seconds"] for x in sub]
            e2e = [x["e2e_seconds"] for x in sub]
            toks = sum(x["total_output_tokens"] for x in sub)
            conds[cname] = {
                "decode_seconds_mean": _mean(dec), "decode_seconds_median": _median(dec),
                "decode_seconds_min": min(dec), "decode_seconds_max": max(dec),
                "prefill_seconds_mean": _mean(pre), "e2e_seconds_mean": _mean(e2e),
                "decode_seconds_sum": sum(dec),
                "aggregate_decode_tps": toks / sum(dec),
                "total_output_tokens": toks,
                "decode_steps_mean": _mean([sum(x["per_seq_global_denoise_steps"]) for x in sub]),
                "reuse_rows": sum(x["counters"].get("reuse_rows", 0) for x in sub),
                "decode_rows": sum(x["counters"].get("decode_rows", 0) for x in sub),
                "update_rows": sum(x["counters"].get("update_rows", 0) for x in sub),
                "store_rows": sum(x["counters"].get("store_rows", 0) for x in sub),
                "saving_rows": sum(x["counters"].get("saving_rows", 0) for x in sub),
                "peak_allocated_mib_max": max(x["peak_allocated_mib"] for x in sub),
                "load_free_min_mib": min(x["load_start"]["mem_free_mib"] for x in sub),
            }
            conds[cname]["reuse_fraction_of_decode_rows"] = (
                conds[cname]["reuse_rows"] / conds[cname]["decode_rows"]
                if conds[cname]["decode_rows"] else None)
        summary["settings"][name] = {
            "batch": s["batch"], "length": s["length"], "pairs": pairs,
            "paired_speedup_mean": _mean(ratios), "paired_speedup_median": _median(ratios),
            "paired_speedup_ci95": _bootstrap_ci(
                ratios, bcfg["reps"], bcfg["seed"], bcfg["confidence"]),
            "conditions": conds,
            "speedup_prefill_adjusted_e2e": (
                conds["dense"]["e2e_seconds_mean"] / conds["s2"]["e2e_seconds_mean"]),
        }
    p = summary["settings"][cfg["primary_setting"]]
    summary["primary"] = {
        "setting": cfg["primary_setting"],
        "mean_paired_decode_speedup": p["paired_speedup_mean"],
        "median_paired_decode_speedup": p["paired_speedup_median"],
        "paired_bootstrap_ci95": p["paired_speedup_ci95"],
        "meets_1_10": bool(p["paired_speedup_mean"] >= 1.10),
        "dense_decode_seconds_mean": p["conditions"]["dense"]["decode_seconds_mean"],
        "s2_decode_seconds_mean": p["conditions"]["s2"]["decode_seconds_mean"],
        "dense_tps": p["conditions"]["dense"]["aggregate_decode_tps"],
        "s2_tps": p["conditions"]["s2"]["aggregate_decode_tps"],
        "reuse_fraction_s2": p["conditions"]["s2"]["reuse_fraction_of_decode_rows"],
        "step_counts_dense": p["conditions"]["dense"]["decode_steps_mean"],
        "step_counts_s2": p["conditions"]["s2"]["decode_steps_mean"],
    }
    table = ["setting,batch,length,dense_decode_s_mean,s2_decode_s_mean,paired_speedup_mean,ci95_lo,ci95_hi,dense_tps,s2_tps,reuse_frac_s2"]
    for s in cfg["settings"]:
        x = summary["settings"][s["name"]]
        table.append(",".join(str(v) for v in [
            s["name"], s["batch"], s["length"],
            round(x["conditions"]["dense"]["decode_seconds_mean"], 6),
            round(x["conditions"]["s2"]["decode_seconds_mean"], 6),
            round(x["paired_speedup_mean"], 6),
            round(x["paired_speedup_ci95"][0], 6), round(x["paired_speedup_ci95"][1], 6),
            round(x["conditions"]["dense"]["aggregate_decode_tps"], 4),
            round(x["conditions"]["s2"]["aggregate_decode_tps"], 4),
            round(x["conditions"]["s2"]["reuse_fraction_of_decode_rows"], 6)]))
    (EXP_DIR / "efficiency-speedup-table.csv").write_text("\n".join(table) + "\n")
    _atomic_json(EXP_DIR / "efficiency-paired-runs.json",
                 {"schema_version": 1, "run_id": RUN_ID,
                  "execution_commit": payload.get("execution_commit"),
                  "rows": rows})
    _atomic_json(EXP_DIR / "efficiency-summary.json", summary)
    metrics = [{"config": name, "seed": 42, "split": "train-only synthetic workload",
                "metric": "mean_paired_dense_decode_seconds/s2_decode_seconds",
                "value": v["paired_speedup_mean"], "unit": "ratio"}
               for name, v in summary["settings"].items()]
    _atomic_json(EXP_DIR / "metrics.json", {"schema_version": 1, "run_id": RUN_ID,
                                            "metrics": metrics})
    print(json.dumps({"status": "SUCCEEDED", "rows": len(rows),
                      "primary": summary["primary"]}))
    return 0


# ---------------------------------------------------------------------------
# resource ledger / finalize
# ---------------------------------------------------------------------------
def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cmd_ledger(args):
    cfg = load_config(args.config)
    prior = float(cfg["budget"]["campaign_before_gpu_hours"])
    total_wall, details = _prior_gpu_wall(include_current_matrix=True)
    files = {}
    for name, w in details.items():
        files[name] = {"wall_seconds": round(w, 3)}
    for p in sorted(RUN_DIR.glob("*.json")):
        if p.name not in ("preflight.json", "matrix.json", "profile.json"):
            continue
        try:
            d = _load(p)
            files[p.name]["status"] = d.get("status")
            files[p.name]["execution_commit"] = d.get("execution_commit")
            files[p.name]["path"] = str(p.relative_to(ROOT))
        except Exception:
            pass
    gpu_hours = total_wall / 3600.0
    ledger = {"schema_version": 1, "run_id": RUN_ID,
              "commands": files,
              "gpu_command_wall_seconds_sum": round(total_wall, 3),
              "gpu_command_wall_hours": round(gpu_hours, 6),
              "campaign_before_gpu_hours": prior,
              "campaign_after_upper_gpu_hours": round(prior + gpu_hours, 6),
              "campaign_limit_gpu_hours": cfg["budget"]["campaign_limit_gpu_hours"],
              "stage_limit_gpu_hours": cfg["budget"]["stage_gpu_hours"],
              "note": "conservative sum of measured GPU-command walls including every "
                      "retained preflight attempt, matrix warmups/timed runs and profiles; "
                      "failed/aborted commands are charged too"}
    for fname in ("preflight.json", "matrix.json", "profile.json"):
        p = RUN_DIR / fname
        if p.exists():
            d = _load(p)
            r = d.get("resources", {})
            ledger[fname] = {"gpu_command_wall_seconds": files[fname]["wall_seconds"],
                             "cumulative_gpu_wall_seconds": d.get("cumulative_gpu_wall_seconds"),
                             "physical_free_min_mib": r.get("mem_free_min_mib"),
                             "physical_used_max_mib": r.get("mem_used_max_mib"),
                             "util_gpu_mean_pct": r.get("util_gpu_mean_pct"),
                             "util_gpu_max_pct": r.get("util_gpu_max_pct"),
                             "samples": r.get("samples"), "breach": r.get("breach")}
    ledger["stage_cumulative_gpu_hours_including_this"] = round(gpu_hours, 6)
    _atomic_json(EXP_DIR / "resource-ledger.json", ledger)
    print(json.dumps({"status": "SUCCEEDED", "wall_seconds": total_wall,
                      "gpu_hours": gpu_hours, "campaign_after": prior + gpu_hours,
                      "files": list(details)}))
    return 0


def cmd_commands(args):
    cfg = load_config(args.config)
    entries = []
    cmds = [
        ("environment_probe", "repro/efficiency_cli.py probe --out experiments/" + RUN_ID + "/environment.json"),
        ("input_freeze", "repro/efficiency_cli.py freeze"),
        ("preflight", "repro/efficiency_cli.py preflight --out runs/" + RUN_ID + "/preflight.json"),
        ("matrix", "repro/efficiency_cli.py matrix"),
        ("profile", "repro/efficiency_cli.py profile --out runs/" + RUN_ID + "/profile.json"),
        ("summary", "repro/efficiency_cli.py summarize"),
        ("ledger", "repro/efficiency_cli.py ledger"),
    ]
    for stage, cmd in cmds:
        entries.append({"stage": stage, "argv": [sys.executable] + cmd.split(),
                        "git_head_at_stage": None, "note": "see per-artifact execution_commit"})
    payload = {"schema_version": 1, "run_id": RUN_ID,
               "task_sha256": _sha256(ROOT / "experiments" / RUN_ID / "task.json")
               if (ROOT / "experiments" / RUN_ID / "task.json").exists() else None,
               "config_sha256": _sha256(CONFIG_PATH),
               "input_manifest_sha256": _sha256(EXP_DIR / "input-manifest.json"),
               "environment": "isolated venv Python3.10 torch2.4.1+cu121 triton3.0 numpy1.26.4 "
                              "transformers4.52.4 FlashInfer0.2.9 custom FlashAttention; no reinstall",
               "entries": entries}
    _atomic_json(EXP_DIR / "commands-and-provenance.json", payload)
    print(json.dumps({"status": "SUCCEEDED", "out": str(EXP_DIR / "commands-and-provenance.json")}))
    return 0


def cmd_finalize(args):
    cfg = load_config(args.config)
    summary = _load(EXP_DIR / "efficiency-summary.json")
    preflight = _load(RUN_DIR / "preflight.json")
    profile = _load(RUN_DIR / "profile.json")
    ledger = _load(EXP_DIR / "resource-ledger.json")
    rows = _load(EXP_DIR / "efficiency-paired-runs.json")["rows"]
    commit_log = subprocess.check_output(
        ["git", "-C", str(ROOT), "log", "--format=%H%x09%s", "-n", "20"],
        text=True).strip().splitlines()
    commits = [{"sha": x.split("\t")[0], "subject": x.split("\t", 1)[1]} for x in commit_log]
    head = _git_head()
    status = "SUCCEEDED" if (summary and preflight.get("status") == "SUCCEEDED"
                             and profile.get("status") == "SUCCEEDED"
                             and summary.get("all_rows") == 60) else "FAILED"
    run_commits = [
        {"stage": "driver_config_manifest", "sha": args.driver_commit,
         "note": "frozen config, input recipe, driver"},
        {"stage": "input_manifest", "sha": args.manifest_commit,
         "note": "deterministic token manifest hashes"},
        {"stage": "preflight_execution", "sha": preflight.get("execution_commit"),
         "note": "actual cached-slot identity preflight execution commit"},
        {"stage": "matrix_execution", "sha": rows[0].get("execution_commit") if rows else None,
         "note": "actual 60-row timing execution commit"},
        {"stage": "profile_execution", "sha": profile.get("execution_commit"),
         "note": "separate profiler execution commit"},
    ]
    report = {
        "schema_version": 1, "run_id": RUN_ID, "status": status,
        "summary": (
            f"Frozen 60-row paired efficiency matrix on one shared RTX3090 with original "
            f"TraDo-4B. Primary B4L8192 mean paired dense/s2 decode speedup "
            f"{summary['primary']['mean_paired_decode_speedup']:.4f} "
            f"(descriptive 95% CI {summary['primary']['paired_bootstrap_ci95'][0]:.4f}-"
            f"{summary['primary']['paired_bootstrap_ci95'][1]:.4f}); "
            f"frozen >=1.10 target met={summary['primary']['meets_1_10']}. "
            f"All 60 rows retained, no reruns/dropouts. Prefill+decode only for pretokenized "
            f"requests; model init/tokenization/detokenization excluded. Shared-GPU load noise "
            f"is reported descriptively, not corrected. No heldout QA or training."),
        "executed_commands": [
            "cd /data/research/flashblock-reproduction",
            "runtime/venv/bin/python repro/efficiency_cli.py probe --out experiments/" + RUN_ID + "/environment.json",
            "runtime/venv/bin/python repro/efficiency_cli.py freeze",
            "runtime/venv/bin/python repro/efficiency_cli.py preflight --out runs/" + RUN_ID + "/preflight.json",
            "runtime/venv/bin/python repro/efficiency_cli.py matrix",
            "runtime/venv/bin/python repro/efficiency_cli.py profile --out runs/" + RUN_ID + "/profile.json",
            "runtime/venv/bin/python repro/efficiency_cli.py summarize",
            "runtime/venv/bin/python repro/efficiency_cli.py ledger",
        ],
        "metrics": [
            {"config": "B4L8192", "seed": 42, "split": "train-only synthetic workload",
             "metric": "mean_paired_dense_decode_seconds/s2_decode_seconds",
             "value": summary["primary"]["mean_paired_decode_speedup"], "unit": "ratio"}
        ] + [
            {"config": s, "seed": 42, "split": "train-only synthetic workload",
             "metric": "mean_paired_dense_decode_seconds/s2_decode_seconds",
             "value": v["paired_speedup_mean"], "unit": "ratio"}
            for s, v in summary["settings"].items() if s != cfg["primary_setting"]
        ],
        "artifacts": ["metrics.json", "environment.json", "input-manifest.json",
                      "preflight.json", "efficiency-paired-runs.json",
                      "efficiency-summary.json", "efficiency-speedup-table.csv",
                      "separate-profile.json", "resource-ledger.json",
                      "commands-and-provenance.json", "report-summary.md"],
        "deviations": [
            "Sequential prefill (add one request, run only its PREFILL, then add next; defer "
            "decode) implemented identically both conditions because Scheduler.schedule does "
            "not enforce max_num_batched_tokens; documented scheduling choice.",
            "Profiler latencies are separate and excluded from the 60 primary rows.",
            "This is an engineering efficiency measurement, not a paper-table replication."
        ],
        "failures": [],
        "resource_usage": {
            "wall_seconds": ledger["gpu_command_wall_seconds_sum"],
            "gpu_hours": ledger["gpu_command_wall_hours"],
            "stage_gpu_hours_limit": cfg["budget"]["stage_gpu_hours"],
            "campaign_before_gpu_hours": cfg["budget"]["campaign_before_gpu_hours"],
            "campaign_after_upper_gpu_hours": ledger["campaign_after_upper_gpu_hours"],
            "note": "conservative sum of measured GPU-command walls including failures",
        },
        "next_recommendation": (
            "GPT should independently recompute the paired statistics from all 60 rows and "
            "inspect identity/preflight and shared-load evidence. Do not start the 128-item "
            "quality stage without a new frozen task."),
        "version_control": {
            "repository": "Flutter-Misdreavus/flashblock-reproduction",
            "visibility": "private",
            "branch": subprocess.check_output(
                ["git", "-C", str(ROOT), "branch", "--show-current"], text=True).strip(),
            "run_commits": run_commits,
            "head_commit": head,
            "commits": commits,
            "push_status": "pending",
            "push_error": "relay mode: executor commits locally only; GPT pulls the private "
                          "repository objects and verifies/pushes",
        },
    }
    _atomic_json(RUN_DIR / "report.json", report)
    print(json.dumps({"status": status, "out": str(RUN_DIR / "report.json"),
                      "head": head}))
    return 0 if status == "SUCCEEDED" else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("freeze"); p.add_argument("--config", default=str(CONFIG_PATH)); p.set_defaults(func=cmd_freeze)
    p = sub.add_parser("probe"); p.add_argument("--config", default=str(CONFIG_PATH)); p.add_argument("--out", required=True); p.set_defaults(func=cmd_probe)
    p = sub.add_parser("preflight"); p.add_argument("--config", default=str(CONFIG_PATH)); p.add_argument("--out", required=True); p.set_defaults(func=cmd_preflight)
    p = sub.add_parser("matrix"); p.add_argument("--config", default=str(CONFIG_PATH)); p.set_defaults(func=cmd_matrix)
    p = sub.add_parser("profile"); p.add_argument("--config", default=str(CONFIG_PATH)); p.add_argument("--out", required=True); p.set_defaults(func=cmd_profile)
    p = sub.add_parser("summarize"); p.add_argument("--config", default=str(CONFIG_PATH)); p.set_defaults(func=cmd_summarize)
    p = sub.add_parser("ledger"); p.add_argument("--config", default=str(CONFIG_PATH)); p.set_defaults(func=cmd_ledger)
    p = sub.add_parser("commands"); p.add_argument("--config", default=str(CONFIG_PATH)); p.set_defaults(func=cmd_commands)
    p = sub.add_parser("finalize"); p.add_argument("--config", default=str(CONFIG_PATH)); p.add_argument("--driver-commit", default="pending"); p.add_argument("--manifest-commit", default="pending"); p.set_defaults(func=cmd_finalize)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
