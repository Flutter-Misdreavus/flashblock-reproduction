#!/usr/bin/env python3
"""Mechanism-gate CLI for run flashblock-20261023-mechanism.

Frozen task: instrument post-RoPE attention inputs/outputs of four selected
TraDo-4B layers/heads on the fixed GSM8K train prompts 100..115, batch 1,
max 512 ordinary-EOS tokens, seed 42, dense (ratio 0) and official s2 (tau=2).

The driver never replaces model outputs and never consumes sampling RNG.
It computes an independent FP32 decomposition of each selected attention head
into (a) block-external (committed prefix) attention and (b) block-internal
(current 4-token block) attention, plus the exact merged reference.

Subcommands
  unit        synthetic numerical preflight: prefill block-causal GQA parity,
              paged KV scatter, FP32 split/recombine, history-page mapping.
  generate    run the real engine for one condition and a list of prompts.
  preflight   unit + instrumented-vs-no-hook same-seed tiny neutrality run.
  summarize   combine dense and s2 results into the mechanism exports.
  finalize    write report.json with the final version-control block.
"""
import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from array import array
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLASH = ROOT / "flash-attention"
DLLM = ROOT / "dllm"
CONFIG_PATH = ROOT / "experiments/flashblock-20261023-mechanism/mechanism-config.json"
RUN_ID = "flashblock-20261023-mechanism"
MASK_TOKEN_ID = 151669
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
PROMPT_TEMPLATE = (
    "<|im_start|>user\nPlease reason step by step, and put your final answer within "
    "\\boxed{{}}.\n{problem}<|im_end|>\n<|im_start|>assistant\n"
)

torch = None
CAPTURE = {
    "enabled": False,
    "collector": None,
    "step": None,
    "hook_counts": Counter(),
    "errors": [],
}


# ---------------------------------------------------------------------------
# small utilities
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
    controls = cfg["controls"]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = controls["gpu_uuid"]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRADO_CONSISTENT_SAMPLING_PARAMS"] = controls["consistent_sampling_params_env"]
    os.environ["TRADO_WARMUP_MODEL_LEN"] = str(controls["warmup_model_len"])
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ.setdefault("OMP_NUM_THREADS", str(controls["cpu_threads"]))
    os.environ.setdefault("MKL_NUM_THREADS", str(controls["cpu_threads"]))
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
    os.environ["FLASH_ATTN_CUDA_ARCHS"] = "86"
    # the isolated venv ships the ninja executable used by flashinfer JIT
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
    return {
        "seed": seed,
        "python_random_first": int(random.getstate()[1][0]),
        "numpy_first": int(np.random.get_state()[1][0]),
        "torch_initial_seed": int(torch.initial_seed()),
        "torch_cuda_initial_seed": int(torch.cuda.initial_seed()),
    }


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


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


def _rel_l2(a, b):
    _init_torch()
    a = a.float()
    b = b.float()
    denom = torch.linalg.vector_norm(b).item()
    return (torch.linalg.vector_norm(a - b).item() / denom) if denom else float("nan")


def _max_abs(a, b):
    _init_torch()
    return (a.float() - b.float()).abs().max().item()


def _rel_l2_rows(a, b, eps=1e-12):
    """Row-wise relative L2 over the last dimension."""
    diff = torch.linalg.vector_norm(a - b, dim=-1)
    den = torch.linalg.vector_norm(b, dim=-1)
    return diff / torch.clamp(den, min=eps)


def _cos_rows(a, b, eps=1e-12):
    """Row-wise cosine and a validity mask for non-zero norms."""
    na = torch.linalg.vector_norm(a, dim=-1)
    nb = torch.linalg.vector_norm(b, dim=-1)
    dot = (a * b).sum(dim=-1)
    valid = (na > eps) & (nb > eps)
    cos = dot / torch.clamp(na * nb, min=eps)
    return cos, valid


def _merge_attn(a1, l1, a2, l2):
    """Exact log-space merge of two attention outputs with LSE [.., 1]."""
    m = torch.maximum(l1, l2)
    w1 = torch.exp(l1 - m)
    w2 = torch.exp(l2 - m)
    den = w1 + w2
    out = (a1 * w1.unsqueeze(-1) + a2 * w2.unsqueeze(-1)) / torch.clamp(den.unsqueeze(-1), min=1e-30)
    lse = m + torch.log(torch.clamp(den, min=1e-30))
    return out, lse


def _free_mem_mib():
    _init_torch()
    free, total = torch.cuda.mem_get_info()
    return free / 2**20, total / 2**20


def _load_gsm8k_rows(parquet, indices):
    import pyarrow.parquet as pq
    rows = pq.read_table(parquet).to_pylist()
    return [dict(rows[i], _index=int(i)) for i in indices]


def _build_prompts(cfg, rows):
    _init_torch()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["local_path"], use_fast=True, trust_remote_code=True)
    prompts, audit = [], []
    for r in rows:
        content = INSTRUCTION + "\n" + r["question"]
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        hard = PROMPT_TEMPLATE.format(problem=r["question"])
        audit.append({
            "index": r["_index"],
            "matches_tokenizer_template": bool(chat == hard),
            "prompt_tokens": len(tokenizer.encode(chat)),
            "prompt_sha256": hashlib.sha256(chat.encode()).hexdigest(),
        })
        prompts.append(chat if chat == hard else hard)
    return tokenizer, prompts, audit


# ---------------------------------------------------------------------------
# independent reference math
# ---------------------------------------------------------------------------
def _gather_history(cache, block_table, num_tokens, kv_heads, page_size):
    """Read committed prefix K/V rows from a paged cache using the TRUE block table.

    Returns [num_tokens, len(kv_heads), head_dim] float32, or None for empty history.
    """
    _init_torch()
    if num_tokens <= 0:
        return None
    if len(block_table) == 0:
        raise RuntimeError("non-empty history but empty block_table")
    bt = torch.as_tensor(block_table, dtype=torch.long, device=cache.device)
    pos = torch.arange(num_tokens, device=cache.device, dtype=torch.long)
    block_idx = torch.div(pos, page_size, rounding_mode="floor")
    offset = pos - block_idx * page_size
    if int(block_idx.max().item()) >= int(bt.numel()):
        raise RuntimeError(
            f"history page mapping out of range: need page {int(block_idx.max())}, "
            f"block_table has {int(bt.numel())} entries")
    phys = bt[block_idx]
    if bool((phys < 0).any().item()):
        raise RuntimeError("negative physical block id in true history block table")
    kh = torch.as_tensor(list(kv_heads), dtype=torch.long, device=cache.device)
    gathered = cache[phys, offset]  # [L, num_kv_heads, D]
    return gathered.index_select(1, kh).float()


def _ref_components(q_sel, k_hist, v_hist, k_cur, v_cur, head_kv_local, scale, eps=1e-12):
    """Independent FP32 external/internal/exact attention for selected heads.

    q_sel: [P,H,D] float32 post-RoPE queries.
    k_hist/v_hist: [L,KVsel,D] float32 committed-prefix K/V, or None.
    k_cur/v_cur: [P,KVsel,D] float32 current-block K/V.
    head_kv_local: list mapping each selected query head to a column of KVsel.
    """
    _init_torch()
    P, H, D = q_sel.shape
    idx = torch.as_tensor(list(head_kv_local), dtype=torch.long, device=q_sel.device)
    k0 = k_cur.index_select(1, idx)
    v0 = v_cur.index_select(1, idx)

    if k_hist is not None and k_hist.shape[0] > 0:
        kh = k_hist.index_select(1, idx)
        vh = v_hist.index_select(1, idx)
        s_ext = torch.einsum("phd,lhd->phl", q_sel, kh) * scale
        l_ext = torch.logsumexp(s_ext, dim=-1)
        p_ext = torch.softmax(s_ext, dim=-1)
        a_ext = torch.einsum("phl,lhd->phd", p_ext, vh)
    else:
        a_ext = torch.zeros((P, H, D), dtype=torch.float32, device=q_sel.device)
        l_ext = torch.full((P, H), float("-inf"), dtype=torch.float32, device=q_sel.device)

    s_in = torch.einsum("phd,jhd->phj", q_sel, k0) * scale
    l_in = torch.logsumexp(s_in, dim=-1)
    p_in = torch.softmax(s_in, dim=-1)
    a_in = torch.einsum("phj,jhd->phd", p_in, v0)

    a_full, l_full = _merge_attn(a_ext, l_ext, a_in, l_in)
    return {
        "a_ext": a_ext, "l_ext": l_ext,
        "a_in": a_in, "l_in": l_in,
        "a_full": a_full, "l_full": l_full,
    }


# ---------------------------------------------------------------------------
# synthetic preflight tests
# ---------------------------------------------------------------------------
def _block_causal_mask(Lq, Lk, stairs):
    _init_torch()
    row = torch.arange(Lq, device="cuda").view(-1, 1)
    col = torch.arange(Lk, device="cuda").view(1, -1)
    log2 = int(math.log2(stairs))
    limit = (((row >> log2) + 1) << log2) - 1
    return col <= limit


def _ref_prefill(q, k, v, cu_q, cu_k, stairs):
    """Independent FP32 block-causal reference for variable-length prefill."""
    _init_torch()
    T, H, D = q.shape
    _, Hk, _ = k.shape
    rep = H // Hk
    scale = D ** -0.5
    out = torch.zeros((T, H, D), dtype=torch.float32, device=q.device)
    cuq = cu_q.cpu().tolist()
    cuk = cu_k.cpu().tolist()
    for s in range(len(cuq) - 1):
        q0, q1 = cuq[s], cuq[s + 1]
        k0, k1 = cuk[s], cuk[s + 1]
        qs = q[q0:q1].float()
        ks = k[k0:k1].float().repeat_interleave(rep, dim=1)
        vs = v[k0:k1].float().repeat_interleave(rep, dim=1)
        scores = torch.einsum("qhd,khd->hqk", qs, ks) * scale
        mask = _block_causal_mask(q1 - q0, k1 - k0, stairs)
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[q0:q1] = torch.einsum("hqk,khd->qhd", probs, vs)
    return out


def run_unit_tests(cfg):
    _init_torch()
    from jetengine_ext.layers.attention import store_kvcache
    from jetengine_ext.kernels.triton.attention import sparse_attn_varlen
    torch.manual_seed(0)
    results = {}

    # 1) prefill block-causal GQA parity, lengths 75 and 128
    try:
        torch.manual_seed(11)
        Ls = [75, 128]
        T = sum(Ls)
        H, Hk, D, stairs = 32, 8, 128, 4
        q = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(T, Hk, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(T, Hk, D, device="cuda", dtype=torch.bfloat16)
        cu = torch.tensor([0, Ls[0], T], device="cuda", dtype=torch.int32)
        o = sparse_attn_varlen(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                               staircase_size=stairs)
        ref = _ref_prefill(q, k, v, cu, cu, stairs)
        rl = _rel_l2(o, ref)
        ma = _max_abs(o, ref)
        results["prefill_block_causal_gqa"] = {
            "lengths": Ls, "heads": H, "kv_heads": Hk, "head_dim": D,
            "rel_l2": rl, "max_abs": ma,
            "pass": bool(rl <= cfg["tolerances"]["prefill_bf16_rel_l2_max"]
                         and ma <= cfg["tolerances"]["prefill_bf16_max_abs_max"]),
        }
    except Exception as e:
        results["prefill_block_causal_gqa"] = {"pass": False, "error": repr(e)}

    # 2) repaired paged KV scatter vs direct indexing, non-contiguous and negative slots
    try:
        num_blocks, page, nkv, d = 4, 256, 8, 128
        k_cache = torch.full((num_blocks, page, nkv, d), 7.0, device="cuda", dtype=torch.bfloat16)
        v_cache = torch.full((num_blocks, page, nkv, d), -3.0, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(7, nkv, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(7, nkv, d, device="cuda", dtype=torch.bfloat16)
        slots = torch.tensor([5, -1, 260, 3, 0, -1, 511], device="cuda", dtype=torch.int32)
        store_kvcache(k, v, k_cache, v_cache, slots)
        torch.cuda.synchronize()
        direct_k = torch.full_like(k_cache, 7.0)
        direct_v = torch.full_like(v_cache, -3.0)
        for i in range(7):
            s = int(slots[i].item())
            if s < 0:
                continue
            b, off = s // page, s % page
            direct_k[b, off] = k[i]
            direct_v[b, off] = v[i]
        k_ok = bool(torch.equal(k_cache, direct_k))
        v_ok = bool(torch.equal(v_cache, direct_v))
        # a slot that no positive index targets must remain the sentinel; together
        # with the exact direct-index comparison this proves negative slots are skipped
        slot_last_block, slot_last_off = 1023 // page, 1023 % page
        neg_ok = bool(torch.equal(
            k_cache[slot_last_block, slot_last_off],
            torch.full((nkv, d), 7.0, device="cuda", dtype=torch.bfloat16)))
        neg_ok = neg_ok and bool(torch.equal(
            v_cache[slot_last_block, slot_last_off],
            torch.full((nkv, d), -3.0, device="cuda", dtype=torch.bfloat16)))
        results["paged_kv_scatter"] = {
            "slots": [int(x) for x in slots.tolist()],
            "bitwise_k": k_ok, "bitwise_v": v_ok, "negative_slot_untouched": neg_ok,
            "pass": bool(k_ok and v_ok and neg_ok),
        }
    except Exception as e:
        results["paged_kv_scatter"] = {"pass": False, "error": repr(e)}

    # 3) FP32 split/recombine exactness, including an empty-history block
    try:
        torch.manual_seed(12)
        P, H, KV, D = 4, 4, 4, 128
        scale = D ** -0.5
        q = torch.randn(P, H, D, device="cuda")
        k0 = torch.randn(P, KV, D, device="cuda")
        v0 = torch.randn(P, KV, D, device="cuda")
        parts_empty = _ref_components(q, None, None, k0, v0, [0, 1, 2, 3], scale)
        direct_empty = torch.softmax(torch.einsum("phd,jhd->phj", q, k0) * scale, dim=-1)
        direct_empty = torch.einsum("phj,jhd->phd", direct_empty, v0)
        e_rl = _rel_l2(parts_empty["a_full"], direct_empty)
        e_ma = _max_abs(parts_empty["a_full"], direct_empty)

        L = 64
        kh = torch.randn(L, KV, D, device="cuda")
        vh = torch.randn(L, KV, D, device="cuda")
        parts = _ref_components(q, kh, vh, k0, v0, [0, 1, 2, 3], scale)
        kk = torch.cat([kh, k0], dim=0)
        vv = torch.cat([vh, v0], dim=0)
        p = torch.softmax(torch.einsum("phd,khd->phk", q, kk) * scale, dim=-1)
        direct = torch.einsum("phk,khd->phd", p, vv)
        rl = _rel_l2(parts["a_full"], direct)
        ma = _max_abs(parts["a_full"], direct)
        tol = cfg["tolerances"]
        results["fp32_split_recombine"] = {
            "nonempty_rel_l2": rl, "nonempty_max_abs": ma,
            "empty_history_rel_l2": e_rl, "empty_history_max_abs": e_ma,
            "pass": bool(rl <= tol["fp32_split_recombine_rel_l2_max"]
                         and ma <= tol["fp32_split_recombine_max_abs_max"]
                         and e_rl <= tol["fp32_split_recombine_rel_l2_max"]
                         and e_ma <= tol["fp32_split_recombine_max_abs_max"]),
        }
    except Exception as e:
        results["fp32_split_recombine"] = {"pass": False, "error": repr(e)}

    # 4) reference history-page mapping with a deliberately non-contiguous table
    try:
        num_blocks, page, nkv, d = 5, 256, 8, 128
        cache = torch.zeros((num_blocks, page, nkv, d), device="cuda")
        for blk in range(num_blocks):
            for off in range(page):
                cache[blk, off] = float(blk * 100000 + off)
        table = [3, 0, 2]  # non-contiguous
        L = 600
        got = _gather_history(cache, table, L, [0, 2, 4, 6], page)
        exp = torch.zeros((L, 4, d), device="cuda")
        for j in range(L):
            b, off = j // page, j % page
            exp[j] = cache[table[b], off, [0, 2, 4, 6]]
        ok = bool(torch.equal(got, exp.float()))
        results["history_page_mapping"] = {
            "block_table": table, "num_tokens": L, "exact": ok, "pass": ok,
        }
    except Exception as e:
        results["history_page_mapping"] = {"pass": False, "error": repr(e)}

    torch.cuda.synchronize()
    all_pass = all(v.get("pass", False) for v in results.values())
    return {"status": "SUCCEEDED" if all_pass else "FAILED",
            "all_pass": all_pass, "tests": results,
            "execution_commit": _git_head(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0)}


# ---------------------------------------------------------------------------
# resource sampler
# ---------------------------------------------------------------------------
class ResourceSampler:
    def __init__(self, interval=0.5, min_free_mib=3072, max_alloc_mib=16384):
        self.interval = interval
        self.min_free_mib = float(min_free_mib)
        self.max_alloc_mib = float(max_alloc_mib)
        self.stop = threading.Event()
        self.free = []
        self.breach = None
        self.thread = None

    def _record_breach(self, kind, value, limit):
        if self.breach is None:
            self.breach = {"kind": kind, "value": float(value), "limit": float(limit),
                           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def check(self):
        _init_torch()
        if self.breach is not None:
            raise RuntimeError(f"resource guard breach: {self.breach}")
        peak = torch.cuda.max_memory_allocated() / 2**20
        if peak > self.max_alloc_mib:
            self._record_breach("peak_allocated_mib", peak, self.max_alloc_mib)
            raise RuntimeError(f"resource guard breach: {self.breach}")

    def start(self):
        _init_torch()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        def _loop():
            while not self.stop.is_set():
                try:
                    free, _ = _free_mem_mib()
                    alloc = torch.cuda.memory_allocated() / 2**20
                    self.free.append(float(free))
                    if free < self.min_free_mib:
                        self._record_breach("free_mib", free, self.min_free_mib)
                    if alloc > self.max_alloc_mib:
                        self._record_breach("allocated_mib", alloc, self.max_alloc_mib)
                except Exception:
                    pass
                time.sleep(self.interval)

        self.thread = threading.Thread(target=_loop, daemon=True)
        self.thread.start()

    def stop_and_summary(self):
        _init_torch()
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
        free_end = None
        try:
            free_end, total = _free_mem_mib()
        except Exception:
            total = None
        out = {
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
            "min_free_mib_sampled": round(min(self.free), 1) if self.free else None,
            "free_mib_start": round(self.free[0], 1) if self.free else None,
            "free_mib_end": round(free_end, 1) if free_end is not None else None,
            "gpu_total_mib": round(total, 1) if total is not None else None,
            "samples": len(self.free),
            "breach": self.breach,
            "limits": {"min_free_mib": self.min_free_mib,
                       "max_allocated_mib": self.max_alloc_mib},
        }
        return out


# ---------------------------------------------------------------------------
# per-prompt collector
# ---------------------------------------------------------------------------
METRICS = [
    "cos_out", "cos_in", "gap", "gap_changed", "gap_unchanged",
    "lse_out_drift", "lse_in_drift", "prev_stale_rel",
    "actual_full_refresh", "actual_full_reuse",
    "stale_merge_rel", "stale_vs_actual", "cache_label_rel", "cache_label_lse_abs",
]
TRACE_PAIR_MAX = 1200
TRACE_STEP_MAX = 400


def _arr_stats(a):
    if len(a) == 0:
        return None
    v = np.frombuffer(a, dtype=np.float32).astype(np.float64)
    finite = np.isfinite(v)
    vf = v[finite]
    if vf.size == 0:
        return {"n": 0, "nonfinite": int((~finite).sum())}
    return {
        "n": int(vf.size),
        "nonfinite": int((~finite).sum()),
        "mean": float(vf.mean()),
        "p10": float(np.percentile(vf, 10)),
        "p50": float(np.percentile(vf, 50)),
        "p90": float(np.percentile(vf, 90)),
        "min": float(vf.min()),
        "max": float(vf.max()),
    }


def _mean_or_none(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


class PromptCollector:
    def __init__(self, cfg, prompt_index, condition):
        self.cfg = cfg
        self.prompt_index = int(prompt_index)
        self.condition = condition
        self.layers = list(cfg["instrumentation"]["layers"])
        self.heads = list(cfg["instrumentation"]["heads"])
        self.acc = {}
        for layer in self.layers:
            for head in self.heads:
                self.acc[(layer, head)] = {m: array("f") for m in METRICS}
        self.prev = {}
        self.age_block = None
        self.last_store_step = None
        self.step_counts = Counter()
        self.cache_age = array("i")
        self.dirty_count = array("i")
        self.n_zero_norm = 0
        self.reuse_by_block = Counter()
        self.pair_trace = []
        self.step_trace = []
        self.hook_counts = Counter()
        self.output = None
        self.meta = {}

    def count_step(self, meta):
        self.step_counts["denoise_total"] += 1
        if meta["status"] == "SAVING":
            self.step_counts["saving_steps"] += 1
        else:
            self.step_counts["denoise_steps"] += 1
        if meta["reuse"]:
            self.step_counts["reuse_steps"] += 1
            self.reuse_by_block[meta["block_start"]] += 1
        else:
            self.step_counts["refresh_steps"] += 1
        if meta["kv_write"]:
            self.step_counts["kv_write_steps"] += 1
        if meta["al_store"]:
            self.step_counts["al_store_steps"] += 1

        if meta["block_start"] != self.age_block:
            self.age_block = meta["block_start"]
            self.last_store_step = None
        if self.last_store_step is None:
            meta["cache_age"] = None
        else:
            meta["cache_age"] = int(meta["step"] - self.last_store_step)
        if meta["al_store"]:
            self.last_store_step = int(meta["step"])

        if meta["status"] != "SAVING":
            if meta["cache_age"] is not None:
                self.cache_age.append(int(meta["cache_age"]))
            self.dirty_count.append(int(meta["dirty_count"]))
            if len(self.step_trace) < TRACE_STEP_MAX:
                self.step_trace.append({
                    "prompt": self.prompt_index, "condition": self.condition,
                    "block_start": meta["block_start"], "step": meta["step"],
                    "status": meta["status"], "reuse": bool(meta["reuse"]),
                    "kv_write": bool(meta["kv_write"]), "al_store": bool(meta["al_store"]),
                    "cache_age": meta["cache_age"], "dirty_count": int(meta["dirty_count"]),
                })

    def set_output(self, out, tokenizer):
        ids = [int(x) for x in out["token_ids"]]
        text = out["text"]
        self.output = {
            "token_ids": ids,
            "text": text,
            "new_tokens": len(ids),
            "ids_in_vocab": bool(all(0 <= x < len(tokenizer) for x in ids)),
            "contains_mask_token": bool(MASK_TOKEN_ID in ids),
            "text_nonempty": bool(text.strip()),
            "token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        }

    def observe_step(self, layer, meta, parts, actual, stale_a, stale_l, post_a, post_l):
        _init_torch()
        a_ext, l_ext = parts["a_ext"], parts["l_ext"]
        a_in, l_in = parts["a_in"], parts["l_in"]
        a_full, l_full = parts["a_full"], parts["l_full"]
        P, H = l_ext.shape
        self.hook_counts[layer] += 1
        CAPTURE["hook_counts"][layer] += 1

        # actual model output vs fresh full FP32 reference
        af_rel = _rel_l2_rows(actual, a_full).detach().cpu().numpy()
        key = "actual_full_reuse" if meta["reuse"] else "actual_full_refresh"

        # stale cache merge diagnostics: only genuine s2 reuse steps consume the
        # cached history A/L, so the approximation residual is reported on those.
        sm_rel = sm_act = None
        if self.condition == "s2" and meta["reuse"] and stale_a is not None:
            sm, _ = _merge_attn(stale_a, stale_l, a_in, l_in)
            sm_rel = _rel_l2_rows(sm, a_full).detach().cpu().numpy()
            sm_act = _rel_l2_rows(sm, actual).detach().cpu().numpy()

        # cache label check: after a real history-A/L store, the buffer must equal
        # the independent external reference for the same query positions.
        cl_rel = cl_lse = None
        if post_a is not None and meta["al_store"] and not meta["reuse"]:
            cl_rel_t = _rel_l2_rows(post_a, a_ext)
            cl_lse_t = (post_l - l_ext).abs()
            cl_rel = cl_rel_t.detach().cpu().numpy()
            cl_lse = cl_lse_t.detach().cpu().numpy()
            hard = self.cfg["tolerances"]["cache_label_rel_l2_hard_max"]
            hard_lse = self.cfg["tolerances"]["cache_label_lse_abs_hard_max"]
            if float(np.nanmax(cl_rel)) > hard or float(np.nanmax(cl_lse)) > hard_lse:
                detail = {
                    "type": "cache_label_mismatch", "layer": layer,
                    "max_rel_l2": float(np.nanmax(cl_rel)),
                    "max_lse_abs": float(np.nanmax(cl_lse)),
                    "step": meta["step"], "block_start": meta["block_start"],
                }
                CAPTURE["errors"].append(detail)
                raise RuntimeError(f"hard cache-label identity failure: {detail}")

        prev = self.prev.get(layer)
        if prev is not None and prev["block_start"] != meta["block_start"]:
            prev = None
        pair_vals = None
        if prev is not None and prev["step"] + 1 == meta["step"]:
            cos_out, valid_out = _cos_rows(a_ext, prev["a_ext"])
            cos_in, valid_in = _cos_rows(a_in, prev["a_in"])
            valid = valid_out & valid_in
            self.n_zero_norm += int((~valid).sum().item())
            gap = cos_out - cos_in
            lse_out = (l_ext - prev["l_ext"]).abs()
            lse_in = (l_in - prev["l_in"]).abs()
            ps, _ = _merge_attn(prev["a_ext"], prev["l_ext"], a_in, l_in)
            ps_rel = _rel_l2_rows(ps, a_full)
            changed = torch.as_tensor(
                [int(meta["token_ids"][p]) != int(prev["token_ids"][p]) for p in range(P)],
                dtype=torch.bool, device=a_ext.device)
            pair_vals = {
                "cos_out": cos_out.detach().cpu().numpy(),
                "cos_in": cos_in.detach().cpu().numpy(),
                "gap": gap.detach().cpu().numpy(),
                "lse_out": lse_out.detach().cpu().numpy(),
                "lse_in": lse_in.detach().cpu().numpy(),
                "ps_rel": ps_rel.detach().cpu().numpy(),
                "valid": valid.detach().cpu().numpy(),
                "changed": changed.detach().cpu().numpy(),
            }

        af_np = af_rel
        for p in range(P):
            for hi, head in enumerate(self.heads):
                key_lh = (layer, head)
                a = self.acc[key_lh]
                a[key].append(float(af_np[p, hi]))
                if sm_rel is not None:
                    a["stale_merge_rel"].append(float(sm_rel[p, hi]))
                    a["stale_vs_actual"].append(float(sm_act[p, hi]))
                if cl_rel is not None:
                    a["cache_label_rel"].append(float(cl_rel[p, hi]))
                    a["cache_label_lse_abs"].append(float(cl_lse[p, hi]))
                if pair_vals is not None and bool(pair_vals["valid"][p, hi]):
                    c_out = float(pair_vals["cos_out"][p, hi])
                    c_in = float(pair_vals["cos_in"][p, hi])
                    g = float(pair_vals["gap"][p, hi])
                    a["cos_out"].append(c_out)
                    a["cos_in"].append(c_in)
                    a["gap"].append(g)
                    a["lse_out_drift"].append(float(pair_vals["lse_out"][p, hi]))
                    a["lse_in_drift"].append(float(pair_vals["lse_in"][p, hi]))
                    a["prev_stale_rel"].append(float(pair_vals["ps_rel"][p, hi]))
                    if bool(pair_vals["changed"][p]):
                        a["gap_changed"].append(g)
                    else:
                        a["gap_unchanged"].append(g)
                    if len(self.pair_trace) < TRACE_PAIR_MAX:
                        self.pair_trace.append({
                            "prompt": self.prompt_index, "condition": self.condition,
                            "layer": layer, "head": head,
                            "block_start": meta["block_start"], "step": meta["step"],
                            "position": p, "token_changed": bool(pair_vals["changed"][p]),
                            "reuse": bool(meta["reuse"]), "cache_age": meta["cache_age"],
                            "cos_out": c_out, "cos_in": c_in, "gap": g,
                            "lse_out_drift": float(pair_vals["lse_out"][p, hi]),
                            "prev_stale_rel_l2": float(pair_vals["ps_rel"][p, hi]),
                            "actual_full_rel_l2": float(af_np[p, hi]),
                        })

        self.prev[layer] = {
            "block_start": meta["block_start"], "step": meta["step"],
            "a_ext": a_ext.detach().clone(), "l_ext": l_ext.detach().clone(),
            "a_in": a_in.detach().clone(), "l_in": l_in.detach().clone(),
            "token_ids": list(meta["token_ids"]),
        }

    def finalize(self):
        per_lh = {}
        for layer in self.layers:
            for head in self.heads:
                a = self.acc[(layer, head)]
                per_lh[f"{layer}:{head}"] = {m: _arr_stats(a[m]) for m in METRICS}

        def lh_mean(metric):
            vals = []
            for layer in self.layers:
                for head in self.heads:
                    st = per_lh[f"{layer}:{head}"].get(metric)
                    if st and st.get("mean") is not None:
                        vals.append(st["mean"])
            return _mean_or_none(vals)

        def lh_pool(metric):
            chunks = []
            for layer in self.layers:
                for head in self.heads:
                    a = self.acc[(layer, head)][metric]
                    if len(a):
                        chunks.append(np.frombuffer(a, dtype=np.float32).astype(np.float64))
            if not chunks:
                return None
            return np.concatenate(chunks)

        def layer_means(metric):
            out = {}
            for layer in self.layers:
                vals = []
                for head in self.heads:
                    st = per_lh[f"{layer}:{head}"].get(metric)
                    if st and st.get("mean") is not None:
                        vals.append(st["mean"])
                out[str(layer)] = _mean_or_none(vals)
            return out

        def head_means(metric):
            out = {}
            for head in self.heads:
                vals = []
                for layer in self.layers:
                    st = per_lh[f"{layer}:{head}"].get(metric)
                    if st and st.get("mean") is not None:
                        vals.append(st["mean"])
                out[str(head)] = _mean_or_none(vals)
            return out

        gap_pool = lh_pool("gap")
        gap_quant = None
        if gap_pool is not None and gap_pool.size:
            fin = gap_pool[np.isfinite(gap_pool)]
            if fin.size:
                gap_quant = {q: float(np.percentile(fin, q)) for q in (10, 50, 90)}
        cache_age_stats = None
        if len(self.cache_age):
            v = np.frombuffer(self.cache_age, dtype=np.int32).astype(np.float64)
            cache_age_stats = {
                "n": int(v.size), "mean": float(v.mean()),
                "p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)),
                "max": float(v.max()),
            }
        dirty_stats = None
        if len(self.dirty_count):
            v = np.frombuffer(self.dirty_count, dtype=np.int32).astype(np.float64)
            dirty_stats = {"n": int(v.size), "mean": float(v.mean()),
                           "max": float(v.max())}
        out = {
            "prompt_index": self.prompt_index,
            "condition": self.condition,
            "output": self.output,
            "steps": dict(self.step_counts),
            "reuse_by_block": {str(k): int(v) for k, v in sorted(self.reuse_by_block.items())},
            "hook_counts": {str(k): int(v) for k, v in sorted(self.hook_counts.items())},
            "cache_age": cache_age_stats,
            "dirty_count": dirty_stats,
            "n_zero_norm_gap_excluded": int(self.n_zero_norm),
            "mean_cos_out": lh_mean("cos_out"),
            "mean_cos_in": lh_mean("cos_in"),
            "mean_gap": lh_mean("gap"),
            "mean_gap_changed": lh_mean("gap_changed"),
            "mean_gap_unchanged": lh_mean("gap_unchanged"),
            "mean_lse_out_drift": lh_mean("lse_out_drift"),
            "mean_lse_in_drift": lh_mean("lse_in_drift"),
            "mean_prev_stale_rel_l2": lh_mean("prev_stale_rel"),
            "mean_actual_full_rel_l2_refresh": lh_mean("actual_full_refresh"),
            "mean_actual_full_rel_l2_reuse": lh_mean("actual_full_reuse"),
            "mean_stale_merge_rel_l2": lh_mean("stale_merge_rel") if self.condition == "s2" else None,
            "mean_stale_vs_actual_rel_l2": lh_mean("stale_vs_actual") if self.condition == "s2" else None,
            "mean_cache_label_rel_l2": lh_mean("cache_label_rel"),
            "max_cache_label_rel_l2": _max_arr(self.acc, self.layers, self.heads, "cache_label_rel"),
            "max_cache_label_lse_abs": _max_arr(self.acc, self.layers, self.heads, "cache_label_lse_abs"),
            "gap_quantiles": gap_quant,
            "gap_by_layer": layer_means("gap"),
            "gap_by_head": head_means("gap"),
            "cos_out_by_layer": layer_means("cos_out"),
            "cos_in_by_layer": layer_means("cos_in"),
            "actual_full_reuse_by_layer": layer_means("actual_full_reuse") if self.condition == "s2" else None,
            "stale_merge_rel_by_layer": layer_means("stale_merge_rel") if self.condition == "s2" else None,
            "per_layer_head": per_lh,
            "trace_pair": self.pair_trace,
            "trace_step": self.step_trace,
        }
        return out


def _max_arr(acc, layers, heads, metric):
    vals = []
    for layer in layers:
        for head in heads:
            a = acc[(layer, head)][metric]
            if len(a):
                v = np.frombuffer(a, dtype=np.float32)
                v = v[np.isfinite(v)]
                if v.size:
                    vals.append(float(v.max()))
    return max(vals) if vals else None


# ---------------------------------------------------------------------------
# engine capture / attention hooks
# ---------------------------------------------------------------------------
def _snapshot_seq(seq):
    sp = seq.sparsity_params
    dirty = 0
    if sp is not None:
        dirty = int(sp.get("dirty_token_count", len(sp.get("dirty_tokens", []))))
    return {
        "seq_id": int(seq.seq_id),
        "status": seq.status.name,
        "num_tokens": int(seq.num_tokens),
        "block_table": [int(x) for x in seq.block_table],
        "intermediate": [int(x) for x in seq.intermediate_block_tokens],
        "current_denoising_step": int(seq.current_denoising_step),
        "global_denoising_step": int(seq.global_denoising_step),
        "num_prompt_tokens": int(seq.num_prompt_tokens),
        "dirty_count": dirty,
    }


def install_scheduler_capture(llm):
    sched = llm.scheduler
    orig = sched.schedule

    def wrapped_schedule():
        guard = CAPTURE.get("resource_guard")
        if guard is not None:
            guard.check()
        pre = {id(s): _snapshot_seq(s) for s in sched.running}
        scheduled, run_type = orig()
        info = {"run_type": run_type.name if run_type is not None else None,
                "valid": False, "seq": None, "meta": None}
        if scheduled:
            if len(scheduled) != 1:
                raise RuntimeError(f"mechanism stage requires batch 1, got {len(scheduled)}")
            seq = scheduled[0]
            snap = pre.get(id(seq))
            if snap is None:
                raise RuntimeError("scheduled sequence had no pre-schedule snapshot")
            sp = seq.sparsity_params
            if sp is not None:
                nu = sp.get("need_update_kvcache_idx", [])
                nnu = sp.get("no_need_update_kvcache_idx", [])
                ns = sp.get("need_store_history_idx", [])

                def _has01(x):
                    if torch.is_tensor(x):
                        return bool((x == 0).any().item())
                    return 0 in list(x)

                reuse = _has01(nnu)
                kv_write = _has01(nu)
                al_store = _has01(ns)
                # read the dirty count actually used by the policy after schedule();
                # schedule() resets the count at a new block, so the pre-snapshot value
                # would otherwise be stale at step 0.
                dirty_after = int(sp.get("dirty_token_count",
                                         len(sp.get("dirty_tokens", []))))
            else:
                reuse, kv_write, al_store = False, True, False
                dirty_after = 0
            meta = {
                "seq_id": snap["seq_id"], "status": snap["status"],
                "block_start": snap["num_tokens"],
                "block_table": snap["block_table"],
                "token_ids": snap["intermediate"],
                "step": snap["current_denoising_step"],
                "global_step": snap["global_denoising_step"],
                "dirty_count": dirty_after,
                "reuse": bool(reuse), "kv_write": bool(kv_write),
                "al_store": bool(al_store), "cache_age": None,
                "run_type": run_type.name if run_type is not None else None,
            }
            info["valid"] = True
            info["seq"] = snap
            info["meta"] = meta
            collector = CAPTURE.get("collector")
            if collector is not None and run_type is not None and run_type.name == "DENOISE":
                collector.count_step(meta)
        CAPTURE["step"] = info
        return scheduled, run_type

    sched.schedule = wrapped_schedule


def _selected(t, heads):
    # t is [P, H, D] float32; select the configured heads
    return t.index_select(1, torch.as_tensor(heads, dtype=torch.long, device=t.device))


def install_attention_hooks(llm, cfg):
    layers = list(cfg["instrumentation"]["layers"])
    heads = list(cfg["instrumentation"]["heads"])
    n_heads = int(cfg["instrumentation"]["num_attention_heads"])
    n_kv = int(cfg["instrumentation"]["num_kv_heads"])
    group = n_heads // n_kv
    kv_heads = sorted({h // group for h in heads})
    page = int(cfg["instrumentation"]["kv_page_size"])
    head_kv_local = [kv_heads.index(h // group) for h in heads]
    modules = llm.model_runner.model.model.layers
    for layer in layers:
        mod = modules[layer].self_attn.attn
        orig = mod.forward

        def make_hook(orig_forward, module, layer_idx):
            def hook(q, k, v, sparsity_params={}):
                info = CAPTURE.get("step")
                collector = CAPTURE.get("collector")
                if (not CAPTURE.get("enabled") or collector is None or info is None
                        or not info.get("valid") or info["run_type"] != "DENOISE"
                        or info["meta"]["status"] == "SAVING"):
                    return orig_forward(q, k, v, sparsity_params)
                meta = info["meta"]
                q_in = q.detach().clone()
                k_in = k.detach().clone()
                v_in = v.detach().clone()
                stale_a = module.attn_output_past[0].detach()[:, heads, :].float().clone()
                stale_l = module.logsumexp[0].detach()[:, heads, :].float().squeeze(-1).clone()
                o = orig_forward(q, k, v, sparsity_params)
                try:
                    P = int(meta["token_ids"].__len__())
                    D = int(module.head_dim)
                    H = len(heads)
                    q_sel = _selected(q_in.view(P, -1, D).float(), heads)
                    k_cur = _selected(k_in.view(P, -1, D).float(), kv_heads).float()
                    v_cur = _selected(v_in.view(P, -1, D).float(), kv_heads).float()
                    hist_k = _gather_history(module.k_cache, meta["block_table"],
                                             int(meta["block_start"]), kv_heads, page)
                    hist_v = _gather_history(module.v_cache, meta["block_table"],
                                             int(meta["block_start"]), kv_heads, page) \
                        if hist_k is not None else None
                    parts = _ref_components(q_sel, hist_k, hist_v, k_cur, v_cur,
                                            head_kv_local, module.scale,
                                            cfg["tolerances"]["zero_norm_eps"])
                    actual = _selected(o.view(P, -1, D).float(), heads)
                    post_a = post_l = None
                    if meta["al_store"] and not meta["reuse"]:
                        post_a = module.attn_output_past[0].detach()[:, heads, :].float()
                        post_l = module.logsumexp[0].detach()[:, heads, :].float().squeeze(-1)
                    collector.observe_step(layer_idx, meta, parts, actual,
                                           stale_a, stale_l, post_a, post_l)
                except Exception:
                    import traceback
                    CAPTURE["errors"].append({
                        "type": "reference_exception", "layer": layer_idx,
                        "traceback": traceback.format_exc()[-2000:],
                        "step": meta.get("step"), "block_start": meta.get("block_start"),
                    })
                    raise
                return o
            return hook

        mod.forward = make_hook(orig, mod, layer)


# ---------------------------------------------------------------------------
# parameter audit
# ---------------------------------------------------------------------------
def _params_finite(module, chunk=1 << 24):
    _init_torch()
    total = 0
    nonfinite = 0
    bad = []
    for name, p in module.named_parameters():
        total += p.numel()
        flat = p.detach().view(-1)
        n_bad = 0
        for i in range(0, flat.numel(), chunk):
            part = flat[i:i + chunk]
            n_bad += int((~torch.isfinite(part)).sum().item())
        if n_bad:
            bad.append({"name": name, "nonfinite": n_bad})
            nonfinite += n_bad
    return {"elements": int(total), "nonfinite_elements": int(nonfinite),
            "nonfinite_tensors": bad[:20],
            "tensors": sum(1 for _ in module.parameters())}


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------
def _make_engine(cfg):
    _init_torch()
    from jetengine_ext.llm import LLM
    controls = cfg["controls"]
    return LLM(
        cfg["model"]["local_path"], enforce_eager=True, tensor_parallel_size=1,
        mask_token_id=MASK_TOKEN_ID, block_length=controls["block_length"],
        batch_size=controls["batch_size"],
        gpu_memory_utilization=controls["gpu_memory_utilization"],
        max_model_len=controls["max_model_len"],
        max_num_batched_tokens=controls["max_num_batched_tokens"],
        max_num_seqs=controls["max_num_seqs"],
    )


def cmd_generate(args):
    cfg = load_config(args.config)
    _configure_env(cfg)
    _setup_paths()
    _init_torch()
    controls = cfg["controls"]
    run_dir = Path(args.run_dir).resolve()
    out_path = Path(args.out).resolve()
    raw_dir = Path(args.raw_dir).resolve() if args.raw_dir else out_path.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if run_dir not in out_path.parents and out_path.parent != run_dir:
        # keep all run artifacts inside the assigned run directory
        raise SystemExit(f"output outside run_dir: {out_path}")

    status_clean = _git_status()
    commit = _git_head()
    result = {
        "schema_version": 1, "run_id": RUN_ID, "condition": args.condition,
        "ratio": float(args.ratio), "hooks": bool(args.hooks),
        "indices": [int(x) for x in args.indices], "max_tokens": int(args.max_tokens),
        "seed": int(controls["seed"]), "status": "FAILED",
        "argv": [sys.executable] + sys.argv,
        "execution_commit": commit, "working_tree_clean_at_start": not bool(status_clean),
        "working_tree_status_at_start": status_clean,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if status_clean and not args.allow_dirty:
        raise SystemExit("refusing executable run with dirty worktree: " + status_clean)

    rows = _load_gsm8k_rows(cfg["dataset"]["train_parquet"], args.indices)
    tokenizer, prompts, template_audit = _build_prompts(cfg, rows)
    result["template_audit"] = template_audit
    result["dataset_sha256"] = cfg["dataset"]["train_sha256"]
    result["model_config_sha256"] = cfg["model"]["config_sha256"]

    from jetengine_ext.sampling_params import SamplingParams
    sp = SamplingParams(
        temperature=controls["temperature"], max_tokens=int(args.max_tokens),
        ignore_eos=bool(controls["ignore_eos"]), block_length=controls["block_length"],
        denoising_steps=controls["denoising_steps"],
        dynamic_threshold=controls["dynamic_threshold"],
        topk=controls["top_k"], topp=controls["top_p"],
        remasking_strategy=controls["remasking_strategy"],
    )

    sampler = ResourceSampler(min_free_mib=controls["min_free_mib"],
                              max_alloc_mib=controls["peak_allocated_limit_mib"])
    CAPTURE["resource_guard"] = sampler
    llm = None
    t0 = time.time()
    prompt_results = []
    audit = {"calls": 0, "nonfinite_calls": 0, "max_numel": 0}
    try:
        seed_init = _seed_all(controls["seed"])
        result["seed_before_init"] = seed_init
        sampler.start()
        llm = _make_engine(cfg)
        result["kv_cache"] = {
            "blocks": int(llm.config.num_kvcache_blocks),
            "block_size": int(llm.config.kvcache_block_size),
            "capacity_tokens": int(llm.config.num_kvcache_blocks * llm.config.kvcache_block_size),
            "gpu_memory_utilization": controls["gpu_memory_utilization"],
        }
        kv_tokens = result["kv_cache"]["capacity_tokens"]
        if kv_tokens < 2 * controls["max_model_len"] + 64:
            raise RuntimeError(f"KV capacity {kv_tokens} < required {2 * controls['max_model_len'] + 64}")

        result["finite_parameters"] = _params_finite(llm.model_runner.model)
        if result["finite_parameters"]["nonfinite_elements"] != 0:
            raise RuntimeError("non-finite model parameters")

        orig_run = llm.model_runner.run_model

        def audited_run(*a, **k):
            logits = orig_run(*a, **k)
            audit["calls"] += 1
            audit["max_numel"] = max(audit["max_numel"], int(logits.numel()))
            if not bool(torch.isfinite(logits).all().item()):
                audit["nonfinite_calls"] += 1
                raise RuntimeError(
                    f"non-finite logits at audited call {audit['calls']} "
                    f"(numel={int(logits.numel())}); hard stop")
            return logits

        llm.model_runner.run_model = audited_run
        install_scheduler_capture(llm)
        if args.hooks:
            install_attention_hooks(llm, cfg)
            CAPTURE["enabled"] = True
        else:
            CAPTURE["enabled"] = False

        for pos, (idx, prompt) in enumerate(zip(args.indices, prompts)):
            collector = PromptCollector(cfg, idx, args.condition)
            CAPTURE["collector"] = collector if args.hooks else None
            seed_gen = _seed_all(controls["seed"])
            t_gen = time.time()
            outs = llm.generate_streaming(
                [prompt], sp, max_active=controls["max_active"],
                use_tqdm=False, sparsity_ratio=float(args.ratio))
            torch.cuda.synchronize()
            gen_seconds = time.time() - t_gen
            if len(outs) != 1:
                raise RuntimeError(f"prompt {idx}: expected 1 output, got {len(outs)}")
            out0 = outs[0]
            collector.set_output(out0, tokenizer)
            pr = collector.finalize()
            pr["generation_seconds"] = round(gen_seconds, 3)
            pr["seed_before_generate"] = seed_gen
            raw = {
                "index": idx, "condition": args.condition,
                "token_ids": collector.output["token_ids"],
                "text": collector.output["text"],
                "template_audit": template_audit[pos],
                "per_prompt_metrics": pr,
            }
            _atomic_json(raw_dir / f"prompt-{idx:03d}.raw.json",
                         {"index": idx, "condition": args.condition,
                          "token_ids": collector.output["token_ids"],
                          "text": collector.output["text"]})
            _atomic_json(raw_dir / f"prompt-{idx:03d}.collector.json", pr)
            prompt_results.append(pr)
            CAPTURE["collector"] = None
            result["prompts"] = prompt_results
            _atomic_json(out_path, result)

            if not collector.output["ids_in_vocab"]:
                raise RuntimeError(f"prompt {idx}: token id out of vocab")
            if collector.output["contains_mask_token"]:
                raise RuntimeError(f"prompt {idx}: mask token in output")
            if not collector.output["text_nonempty"] or collector.output["new_tokens"] <= 0:
                raise RuntimeError(f"prompt {idx}: empty output")
            if args.hooks and sum(collector.hook_counts.values()) == 0:
                raise RuntimeError(f"prompt {idx}: hooks recorded no steps")
            print(json.dumps({"prompt": idx, "condition": args.condition,
                              "tokens": collector.output["new_tokens"],
                              "gap": pr["mean_gap"], "sec": round(gen_seconds, 2)}),
                  flush=True)

        result["logits_audit"] = audit
        result["generation_seconds_total"] = round(time.time() - t0, 3)
        result["hook_counts"] = {str(k): int(v) for k, v in sorted(CAPTURE["hook_counts"].items())}
        result["capture_errors"] = list(CAPTURE["errors"])
        sampler.check()
        if audit["nonfinite_calls"] != 0:
            raise RuntimeError(f"non-finite logits observed ({audit['nonfinite_calls']} calls)")
        if CAPTURE["errors"]:
            raise RuntimeError(f"capture errors observed: {CAPTURE['errors'][:3]}")
        if len(prompt_results) != len(args.indices):
            raise RuntimeError("not all requested prompts produced a trajectory")
        result["status"] = "SUCCEEDED" if all(
            p["output"]["ids_in_vocab"] and p["output"]["text_nonempty"]
            and p["output"]["new_tokens"] > 0 and not p["output"]["contains_mask_token"]
            for p in prompt_results) else "FAILED"
    except Exception as e:
        import traceback
        result["error"] = repr(e)
        result["traceback"] = traceback.format_exc()[-4000:]
        result["status"] = "FAILED"
    finally:
        CAPTURE["enabled"] = False
        CAPTURE["collector"] = None
        try:
            result["resources"] = sampler.stop_and_summary()
        except Exception as e:
            result["resources"] = {"error": repr(e)}
        res = result["resources"]
        limits = {"peak_allocated_mib": controls["peak_allocated_limit_mib"],
                  "min_free_mib": controls["min_free_mib"]}
        if (res.get("breach") is not None
                or (res.get("peak_allocated_mib") is not None
                    and res["peak_allocated_mib"] > limits["peak_allocated_mib"])
                or (res.get("min_free_mib_sampled") is not None
                    and res["min_free_mib_sampled"] < limits["min_free_mib"])):
            result["status"] = "FAILED"
            result["resource_guard_violation"] = {
                "breach": res.get("breach"), "limits": limits,
                "peak_allocated_mib": res.get("peak_allocated_mib"),
                "min_free_mib_sampled": res.get("min_free_mib_sampled"),
            }
        result["wall_seconds"] = round(time.time() - t0, 3)
        result["ended_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result["prompts"] = prompt_results
        result["logits_audit"] = audit if "logits_audit" not in result else result["logits_audit"]
        try:
            if llm is not None:
                llm.exit()
                result["engine_exit"] = "ok"
        except Exception as e:
            result["engine_exit"] = f"error: {e!r}"
        _atomic_json(out_path, result)
    print(json.dumps({"status": result["status"], "condition": args.condition,
                      "out": str(out_path), "error": result.get("error")}))
    return 0 if result["status"] == "SUCCEEDED" else 1


# ---------------------------------------------------------------------------
# unit / preflight commands
# ---------------------------------------------------------------------------
def cmd_unit(args):
    cfg = load_config(args.config)
    _configure_env(cfg)
    _setup_paths()
    _init_torch()
    res = run_unit_tests(cfg)
    res["argv"] = [sys.executable] + sys.argv
    _atomic_json(Path(args.out).resolve(), res)
    print(json.dumps({"status": res["status"], "out": args.out}))
    return 0 if res["status"] == "SUCCEEDED" else 1


def cmd_preflight(args):
    cfg = load_config(args.config)
    _configure_env(cfg)
    _setup_paths()
    _init_torch()
    run_dir = Path(args.run_dir).resolve()
    pf_dir = run_dir / "preflight"
    pf_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    script = str(Path(__file__).resolve())
    base = [py, script]

    def _run(name, cmd, timeout=1500):
        log = pf_dir / f"{name}.log"
        with open(log, "w", encoding="utf-8") as f:
            p = subprocess.run(cmd, cwd=str(ROOT), env=dict(os.environ),
                               stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        return p.returncode, log

    out = {"schema_version": 1, "run_id": RUN_ID, "status": "FAILED",
           "argv": [sys.executable] + sys.argv,
           "execution_commit": _git_head(), "attempt": int(args.attempt),
           "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "unit": None, "neutrality": None, "attempts": []}
    t0 = time.time()
    ok = True
    try:
        unit_path = pf_dir / "unit.json"
        rc, log = _run("unit", base + ["unit", "--config", args.config, "--out", str(unit_path)])
        out["attempts"].append({"name": "unit", "returncode": rc, "log": str(log)})
        if unit_path.exists():
            out["unit"] = json.loads(unit_path.read_text())
        ok = ok and rc == 0 and out["unit"] and out["unit"].get("all_pass", False)

        runs = {}
        for name, hooks in (("nohook", 0), ("hook", 1)):
            outp = pf_dir / f"{name}.json"
            rc, log = _run(name, base + [
                "generate", "--config", args.config, "--run-dir", str(run_dir),
                "--condition", "s2", "--ratio", "2.0", "--hooks", str(hooks),
                "--indices", "100", "--max-tokens", "16",
                "--out", str(outp), "--raw-dir", str(pf_dir / f"{name}-raw")])
            out["attempts"].append({"name": name, "returncode": rc, "log": str(log)})
            if outp.exists():
                runs[name] = json.loads(outp.read_text())
            ok = ok and rc == 0

        if len(runs) == 2:
            def _ids(res):
                return [list(p["output"]["token_ids"]) for p in res.get("prompts", [])]
            ids_a, ids_b = _ids(runs["nohook"]), _ids(runs["hook"])
            hook_res = runs["hook"]
            hook_counts = {}
            n_align = 0
            for p in hook_res.get("prompts", []):
                for k, v in p.get("hook_counts", {}).items():
                    hook_counts[str(k)] = hook_counts.get(str(k), 0) + int(v)
                per_lh = p.get("per_layer_head", {})
                for lh, st in per_lh.items():
                    n_align += int((st.get("gap") or {}).get("n") or 0)
            required_layers = [str(x) for x in cfg["instrumentation"]["layers"]]
            hooks_nonzero = all(hook_counts.get(l, 0) > 0 for l in required_layers)
            out["neutrality"] = {
                "nohook_token_ids_sha256": hashlib.sha256(json.dumps(ids_a).encode()).hexdigest(),
                "hook_token_ids_sha256": hashlib.sha256(json.dumps(ids_b).encode()).hexdigest(),
                "token_ids_equal": bool(ids_a == ids_b),
                "nohook_token_counts": [len(x) for x in ids_a],
                "hook_token_counts": [len(x) for x in ids_b],
                "hook_counts": hook_counts,
                "required_layers": required_layers,
                "all_required_hooks_nonzero": bool(hooks_nonzero),
                "n_aligned_observations": int(n_align),
                "hook_capture_errors": hook_res.get("capture_errors", []),
                "nohook_generation_seconds": runs["nohook"].get("generation_seconds_total"),
                "hook_generation_seconds": hook_res.get("generation_seconds_total"),
                "nohook_resources": runs["nohook"].get("resources"),
                "hook_resources": hook_res.get("resources"),
            }
            ok = ok and out["neutrality"]["token_ids_equal"] and hooks_nonzero and n_align > 0
        else:
            ok = False
        out["status"] = "SUCCEEDED" if ok else "FAILED"
    except Exception as e:
        import traceback
        out["error"] = repr(e)
        out["traceback"] = traceback.format_exc()[-4000:]
        out["status"] = "FAILED"
    out["ended_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out["wall_seconds"] = round(time.time() - t0, 3)
    _atomic_json(Path(args.out).resolve(), out)
    print(json.dumps({"status": out["status"], "out": args.out}))
    return 0 if out["status"] == "SUCCEEDED" else 1


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def _bootstrap_ci(values, reps, seed, confidence):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(reps, v.size))
    means = v[idx].mean(axis=1)
    lo = float(np.percentile(means, (1 - confidence) / 2 * 100))
    hi = float(np.percentile(means, (1 + confidence) / 2 * 100))
    return {"n": int(v.size), "mean": float(v.mean()), "ci_low": lo, "ci_high": hi}


def _prompt_values(prompts, key):
    return [p.get(key) for p in prompts]


def _collect_lh(prompts, metric):
    """Prompt-level layer/head means from per-prompt per_layer_head."""
    out = {}
    for p in prompts:
        for lh, st in p.get("per_layer_head", {}).items():
            s = st.get(metric)
            if s and s.get("mean") is not None:
                out.setdefault(lh, []).append(float(s["mean"]))
    return {lh: {"mean": float(np.mean(v)), "n_prompts": len(v)} for lh, v in out.items()}


def _summarize_condition(cfg, res, label):
    prompts = res.get("prompts", [])
    boot = cfg["bootstrap"]
    gap = _prompt_values(prompts, "mean_gap")
    cos_out = _prompt_values(prompts, "mean_cos_out")
    cos_in = _prompt_values(prompts, "mean_cos_in")
    gap_changed = _prompt_values(prompts, "mean_gap_changed")
    gap_unchanged = _prompt_values(prompts, "mean_gap_unchanged")
    lse_out = _prompt_values(prompts, "mean_lse_out_drift")
    lse_in = _prompt_values(prompts, "mean_lse_in_drift")
    prev_stale = _prompt_values(prompts, "mean_prev_stale_rel_l2")
    af_refresh = _prompt_values(prompts, "mean_actual_full_rel_l2_refresh")
    af_reuse = _prompt_values(prompts, "mean_actual_full_rel_l2_reuse")
    sm_rel = _prompt_values(prompts, "mean_stale_merge_rel_l2")
    sm_act = _prompt_values(prompts, "mean_stale_vs_actual_rel_l2")
    cl_rel = _prompt_values(prompts, "mean_cache_label_rel_l2")
    cl_rel_max = [p.get("max_cache_label_rel_l2") for p in prompts]
    cl_lse_max = [p.get("max_cache_label_lse_abs") for p in prompts]

    lh_gap = _collect_lh(prompts, "gap")
    lh_cos_out = _collect_lh(prompts, "cos_out")
    lh_cos_in = _collect_lh(prompts, "cos_in")
    lh_af_reuse = _collect_lh(prompts, "actual_full_reuse")
    lh_sm = _collect_lh(prompts, "stale_merge_rel")
    gap_lh_values = list(lh_gap.values())
    pos_lh = [v["mean"] for v in gap_lh_values if v["mean"] is not None and v["mean"] > 0]
    neg_lh = [v["mean"] for v in gap_lh_values if v["mean"] is not None and v["mean"] <= 0]

    steps = Counter()
    for p in prompts:
        for k, v in p.get("steps", {}).items():
            steps[k] += int(v)

    def _lh_n(metric):
        out = {}
        for p in prompts:
            for lh, st in p.get("per_layer_head", {}).items():
                s = st.get(metric)
                if s and s.get("n") is not None:
                    out[lh] = out.get(lh, 0) + int(s["n"])
        return out

    gap_n = _lh_n("gap")
    af_reuse_n = _lh_n("actual_full_reuse")
    sm_reuse_n = _lh_n("stale_merge_rel")
    denominators = {
        "bootstrap_units": "prompts (not query rows or tokens)",
        "n_prompt_units_for_primary": int(sum(1 for x in gap if x is not None and np.isfinite(x))),
        "gap_aligned_row_observations_sum_over_layer_heads": int(sum(gap_n.values())),
        "gap_observations_per_layer_head": gap_n,
        "reuse_steps_total": int(steps.get("reuse_steps", 0)),
        "actual_full_reuse_row_observations_sum_over_layer_heads": int(sum(af_reuse_n.values())),
        "actual_full_reuse_observations_per_layer_head": af_reuse_n,
        "stale_merge_reuse_row_observations_sum_over_layer_heads": int(sum(sm_reuse_n.values())),
        "stale_merge_reuse_observations_per_layer_head": sm_reuse_n,
        "denominator_note": "cosine/LSE/error means are equal-weight over selected layer/head means; "
                            "the n fields are raw aligned row observations and are never treated as "
                            "independent bootstrap units.",
    }

    out = {
        "condition": label,
        "n_prompts": len(prompts),
        "n_prompts_expected": len(cfg["dataset"]["indices"]),
        "primary_mean_gap": _bootstrap_ci(gap, boot["reps"], boot["seed"], boot["confidence"]),
        "mean_cos_out": _bootstrap_ci(cos_out, boot["reps"], boot["seed"], boot["confidence"]),
        "mean_cos_in": _bootstrap_ci(cos_in, boot["reps"], boot["seed"], boot["confidence"]),
        "mean_gap_changed_stratum": _bootstrap_ci(gap_changed, boot["reps"], boot["seed"], boot["confidence"]),
        "mean_gap_unchanged_stratum": _bootstrap_ci(gap_unchanged, boot["reps"], boot["seed"], boot["confidence"]),
        "mean_lse_out_abs_drift": _bootstrap_ci(lse_out, boot["reps"], boot["seed"], boot["confidence"]),
        "mean_lse_in_abs_drift": _bootstrap_ci(lse_in, boot["reps"], boot["seed"], boot["confidence"]),
        "mean_prev_step_stale_merge_rel_l2": _bootstrap_ci(prev_stale, boot["reps"], boot["seed"], boot["confidence"]),
        "mean_actual_full_rel_l2_refresh": _bootstrap_ci(af_refresh, boot["reps"], boot["seed"], boot["confidence"]),
        "mean_actual_full_rel_l2_reuse": _bootstrap_ci(af_reuse, boot["reps"], boot["seed"], boot["confidence"]),
        "mean_stale_merge_rel_l2_reuse": _bootstrap_ci(sm_rel, boot["reps"], boot["seed"], boot["confidence"]),
        "mean_stale_vs_actual_rel_l2": _bootstrap_ci(sm_act, boot["reps"], boot["seed"], boot["confidence"]),
        "mean_cache_label_rel_l2": _bootstrap_ci(cl_rel, boot["reps"], boot["seed"], boot["confidence"]),
        "max_cache_label_rel_l2": max([x for x in cl_rel_max if x is not None], default=None),
        "max_cache_label_lse_abs": max([x for x in cl_lse_max if x is not None], default=None),
        "steps": dict(steps),
        "denominators": denominators,
        "gap_by_layer_head": lh_gap,
        "cos_out_by_layer_head": lh_cos_out,
        "cos_in_by_layer_head": lh_cos_in,
        "actual_full_reuse_by_layer_head": lh_af_reuse if label == "s2" else None,
        "stale_merge_rel_by_layer_head": lh_sm if label == "s2" else None,
        "n_layer_head_positive_gap": len(pos_lh),
        "n_layer_head_nonpositive_gap": len(neg_lh),
        "layer_head_gap_min": min([v["mean"] for v in gap_lh_values if v["mean"] is not None], default=None),
        "layer_head_gap_max": max([v["mean"] for v in gap_lh_values if v["mean"] is not None], default=None),
    }
    out["prompt_mean_gaps"] = {str(p["prompt_index"]): p.get("mean_gap") for p in prompts}
    return out


def cmd_summarize(args):
    cfg = load_config(args.config)
    _init_torch()
    run_dir = Path(args.run_dir).resolve()
    dense = json.loads(Path(args.dense).read_text())
    s2 = json.loads(Path(args.s2).read_text())
    out_dir = run_dir / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_prompt = []
    for res in (dense, s2):
        for p in res.get("prompts", []):
            row = {k: v for k, v in p.items() if k not in ("trace_pair", "trace_step")}
            if isinstance(row.get("output"), dict):
                o = row["output"]
                row["output"] = {k: o.get(k) for k in (
                    "new_tokens", "ids_in_vocab", "contains_mask_token",
                    "text_nonempty", "token_ids_sha256")}
            per_prompt.append(row)
    _atomic_json(out_dir / "mechanism-per-prompt.json", per_prompt)

    dense_sum = _summarize_condition(cfg, dense, "baseline_dense")
    s2_sum = _summarize_condition(cfg, s2, "official_s2_tau2")
    primary = dense_sum["primary_mean_gap"]
    ci_low = primary["ci_low"]
    decision = {
        "rule": "descriptive support if mean paired gap > 0; CI entirely > 0 adds evidence only on this small subset",
        "mean_gap_positive": bool(primary["mean"] is not None and primary["mean"] > 0),
        "ci_entirely_positive": bool(ci_low is not None and ci_low > 0),
        "negative_gap_layer_heads_retained": int(dense_sum["n_layer_head_nonpositive_gap"]),
        "heterogeneity_reported": True,
        "layer_head_gap_range": [dense_sum["layer_head_gap_min"],
                                 dense_sum["layer_head_gap_max"]],
        "no_exactness_or_quality_inferred_from_cosine": True,
    }
    summary = {
        "schema_version": 1, "run_id": RUN_ID,
        "primary_condition": "baseline_dense",
        "primary_metric": "per-prompt mean aligned adjacent-step A_out cosine minus A_in cosine, equally averaged over layers/heads",
        "primary": primary,
        "decision": decision,
        "dense": dense_sum,
        "s2": s2_sum,
        "dense_validation": _validation_block(dense),
        "s2_validation": _validation_block(s2),
        "exclusions": {
            "prefill_steps_instrumented": False,
            "saving_steps_excluded": True,
            "block_transitions_excluded_by_same_block_start": True,
            "zero_norm_cos_excluded": True,
            "note": "Saving steps are counted but not used for adjacent cosine pairs.",
        },
        "limitations": [
            "16 fixed GSM8K train prompts, one seed, no heldout/test evaluation.",
            "Cosine and LSE are mechanistic diagnostics, not exactness or answer-quality evidence.",
            "s2 reuse error is measured only on the frozen small subset and fixed sampler.",
        ],
    }
    _atomic_json(out_dir / "mechanism-summary.json", summary)

    # bounded trace (no raw generated text)
    trace = {
        "schema_version": 1, "run_id": RUN_ID,
        "note": "bounded small trace; raw generations remain outside Git in the run directory",
        "dense_pair": dense.get("prompts", [{}])[0].get("trace_pair", [])[:TRACE_PAIR_MAX] if dense.get("prompts") else [],
        "dense_step": dense.get("prompts", [{}])[0].get("trace_step", [])[:TRACE_STEP_MAX] if dense.get("prompts") else [],
        "s2_pair": s2.get("prompts", [{}])[0].get("trace_pair", [])[:TRACE_PAIR_MAX] if s2.get("prompts") else [],
        "s2_step": s2.get("prompts", [{}])[0].get("trace_step", [])[:TRACE_STEP_MAX] if s2.get("prompts") else [],
    }
    # gather a few traces from distinct prompts, still bounded
    for res, prefix in ((dense, "dense"), (s2, "s2")):
        for p in res.get("prompts", []):
            if len(trace[prefix + "_pair"]) >= 2 * TRACE_PAIR_MAX:
                break
            trace[prefix + "_pair"].extend(p.get("trace_pair", [])[:200])
            trace[prefix + "_step"].extend(p.get("trace_step", [])[:60])
    _atomic_json(out_dir / "bounded-small-trace.json", trace)

    # exact plotting data: per condition/prompt/layer/head
    csv_lines = ["condition,prompt,layer,head,mean_cos_out,mean_cos_in,mean_gap,"
                 "n_gap,mean_lse_out_drift,mean_actual_full_rel_l2_refresh,"
                 "mean_actual_full_rel_l2_reuse,mean_stale_merge_rel_l2,"
                 "mean_prev_stale_rel_l2"]
    for res, cond in ((dense, "dense"), (s2, "s2")):
        for p in res.get("prompts", []):
            for lh, st in p.get("per_layer_head", {}).items():
                layer, head = lh.split(":")
                def g(metric, field="mean"):
                    s = st.get(metric)
                    return "" if not s or s.get(field) is None else s.get(field)
                csv_lines.append(",".join(str(x) for x in [
                    cond, p["prompt_index"], layer, head,
                    g("cos_out"), g("cos_in"), g("gap"), g("gap", "n"),
                    g("lse_out_drift"), g("actual_full_refresh"),
                    g("actual_full_reuse"), g("stale_merge_rel"),
                    g("prev_stale_rel"),
                ]))
    (out_dir / "mechanism-figure-data.csv").write_text("\n".join(csv_lines) + "\n")

    # resource ledger
    def _res(x):
        return x.get("resources", {})

    preflight = _read_optional(run_dir / "preflight.json") or {}
    dev_probes = _read_optional(CONFIG_PATH.parent / "development-probes.json") or {}
    formal_wall = (dense.get("wall_seconds") or 0) + (s2.get("wall_seconds") or 0)
    preflight_wall = preflight.get("wall_seconds") or 0
    dev_wall = dev_probes.get("total_conservative_wall_seconds") or 0
    stage_wall = float(formal_wall) + float(preflight_wall) + float(dev_wall)
    peaks = [x for x in (
        _res(dense).get("peak_allocated_mib"), _res(s2).get("peak_allocated_mib"),
        ((preflight.get("neutrality") or {}).get("hook_resources") or {}).get("peak_allocated_mib"),
        ((preflight.get("neutrality") or {}).get("nohook_resources") or {}).get("peak_allocated_mib"),
    ) if x is not None]
    frees = [x for x in (
        _res(dense).get("min_free_mib_sampled"), _res(s2).get("min_free_mib_sampled"),
        ((preflight.get("neutrality") or {}).get("hook_resources") or {}).get("min_free_mib_sampled"),
        ((preflight.get("neutrality") or {}).get("nohook_resources") or {}).get("min_free_mib_sampled"),
    ) if x is not None]
    ledger = {
        "schema_version": 1, "run_id": RUN_ID,
        "accepted_preflight": {
            "status": preflight.get("status"), "attempt": preflight.get("attempt"),
            "wall_seconds": preflight.get("wall_seconds"),
            "execution_commit": preflight.get("execution_commit"),
        },
        "development_probes": dev_probes,
        "dense": {"wall_seconds": dense.get("wall_seconds"), "resources": _res(dense),
                  "execution_commit": dense.get("execution_commit"),
                  "status": dense.get("status")},
        "s2": {"wall_seconds": s2.get("wall_seconds"), "resources": _res(s2),
               "execution_commit": s2.get("execution_commit"),
               "status": s2.get("status")},
        "formal_wall_seconds": round(formal_wall, 3),
        "accepted_preflight_wall_seconds": round(float(preflight_wall), 3),
        "development_probe_wall_seconds_charged": round(float(dev_wall), 3),
        "total_stage_wall_seconds_conservative": round(stage_wall, 3),
        "conservative_gpu_hours_this_stage": round(stage_wall / 3600.0, 6),
        "campaign_gpu_hours_before": cfg["budget"]["campaign_gpu_hours_before"],
        "campaign_gpu_hours_after_conservative": round(
            cfg["budget"]["campaign_gpu_hours_before"] + stage_wall / 3600.0, 6),
        "peak_allocated_mib_max": max(peaks) if peaks else None,
        "min_free_mib_sampled_min": min(frees) if frees else None,
        "note": "GPU hours charged conservatively as whole-stage wall time on one GPU; "
                "includes failed/unaccepted development probes and the accepted preflight, "
                "not only final dense/s2 runs.",
    }
    _atomic_json(out_dir / "resource-ledger.json", ledger)

    def _cmd_from_argv(res, fallback):
        argv = res.get("argv")
        if argv:
            return " ".join(str(x) for x in argv)
        return fallback

    commands = []
    for name, res in (("dense", dense), ("s2", s2)):
        fallback = (f"{{venv_python}} repro/mechanism_cli.py generate --condition {res.get('condition')} "
                    f"--ratio {res.get('ratio')} --hooks {int(bool(res.get('hooks')))} "
                    f"--indices {' '.join(str(x) for x in res.get('indices', []))} "
                    f"--max-tokens {res.get('max_tokens')} --out <run_dir>/{name}/generate-result.json")
        commands.append({
            "stage": "formal_" + name,
            "command": _cmd_from_argv(res, fallback),
            "execution_commit": res.get("execution_commit"),
            "status": res.get("status"),
            "wall_seconds": res.get("wall_seconds"),
        })
    for stage, res in (("accepted_preflight", preflight),):
        commands.append({
            "stage": stage,
            "command": _cmd_from_argv(res, "{venv_python} repro/mechanism_cli.py preflight"),
            "execution_commit": res.get("execution_commit"),
            "status": res.get("status"),
            "wall_seconds": res.get("wall_seconds"),
        })
    for probe in dev_probes.get("probes", []):
        commands.append({
            "stage": "development_probe:" + probe.get("name", "unknown"),
            "command": "uncommitted worktree probe (see development-probes.json)",
            "execution_commit": probe.get("base_commit"),
            "status": probe.get("status"),
            "wall_seconds": probe.get("wall_seconds"),
            "accepted_evidence": False,
        })
    provenance = {
        "schema_version": 1, "run_id": RUN_ID,
        "execution_commits": {"dense": dense.get("execution_commit"),
                              "s2": s2.get("execution_commit"),
                              "preflight": (_read_optional(run_dir / "preflight.json") or {}).get("execution_commit")},
        "commands": commands,
        "config_sha256": _sha256(Path(args.config)),
        "frozen_config": load_config(args.config),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__ if torch is not None else None,
            "cuda": torch.version.cuda if torch is not None else None,
        },
        "git": {"repository": cfg.get("repository", "Flutter-Misdreavus/flashblock-reproduction"),
                "branch": "exp/flashblock-20261023",
                "head_at_summarize": _git_head()},
        "raw_evidence_location": "run directory (outside Git)",
    }
    _atomic_json(out_dir / "commands-and-provenance.json", provenance)

    # report body (final version_control/head added by finalize)
    status = "SUCCEEDED"
    reasons = []
    if dense.get("status") != "SUCCEEDED" or s2.get("status") != "SUCCEEDED":
        status = "FAILED"
        reasons.append("one or both formal conditions did not succeed")
    if dense_sum["n_prompts"] != len(cfg["dataset"]["indices"]) or s2_sum["n_prompts"] != len(cfg["dataset"]["indices"]):
        status = "FAILED"
        reasons.append("not all fixed prompts produced a trajectory")
    if primary["mean"] is None:
        status = "FAILED"
        reasons.append("primary metric unavailable")
    for name, s in (("dense", dense_sum), ("s2", s2_sum)):
        if s.get("max_cache_label_rel_l2") is not None and s["max_cache_label_rel_l2"] > cfg["tolerances"]["cache_label_rel_l2_hard_max"]:
            status = "FAILED"
            reasons.append(f"{name} cache-label reference mismatch")
        af = s.get("mean_actual_full_rel_l2_refresh", {}).get("mean")
        if af is not None and af > 0.05:
            status = "FAILED"
            reasons.append(f"{name} fresh reference does not match model refresh output (relL2={af})")
    metrics = []

    def _add_metric(config, metric, value, unit, **extra):
        if value is None:
            return
        row = {"config": config, "seed": cfg["controls"]["seed"],
               "split": "gsm8k_train_100_115", "metric": metric,
               "value": value, "unit": unit}
        row.update(extra)
        metrics.append(row)

    if primary["mean"] is not None:
        _add_metric("baseline_dense", "mean_paired_Aout_minus_Ain_cosine_gap",
                    primary["mean"], "cosine",
                    ci95_low=primary["ci_low"], ci95_high=primary["ci_high"])
        _add_metric("baseline_dense", "mean_Aout_cosine",
                    dense_sum["mean_cos_out"]["mean"], "cosine")
        _add_metric("baseline_dense", "mean_Ain_cosine",
                    dense_sum["mean_cos_in"]["mean"], "cosine")
        _add_metric("baseline_dense", "mean_aligned_Lout_absolute_drift",
                    dense_sum["mean_lse_out_abs_drift"]["mean"], "nats")
        _add_metric("official_s2_tau2", "mean_actual_s2_vs_fresh_full_rel_l2_reuse_steps",
                    s2_sum["mean_actual_full_rel_l2_reuse"]["mean"], "relative_L2")
        _add_metric("official_s2_tau2", "mean_independent_stale_merge_rel_l2_reuse_steps",
                    s2_sum["mean_stale_merge_rel_l2_reuse"]["mean"], "relative_L2")
    summary_text = (
        f"Dense baseline on {dense_sum['n_prompts']} fixed GSM8K train prompts: mean paired "
        f"A_out-A_in cosine gap={primary['mean']}, prompt-bootstrap 95% CI "
        f"[{primary['ci_low']}, {primary['ci_high']}]; mean cos_out={dense_sum['mean_cos_out']['mean']}, "
        f"mean cos_in={dense_sum['mean_cos_in']['mean']}. "
        f"{dense_sum['n_layer_head_positive_gap']}/{dense_sum['n_layer_head_positive_gap'] + dense_sum['n_layer_head_nonpositive_gap']} "
        f"selected layer-head means are positive (range {dense_sum['layer_head_gap_min']}.."
        f"{dense_sum['layer_head_gap_max']}; heterogeneity retained), with changed-row "
        f"stratum mean {dense_sum['mean_gap_changed_stratum']['mean']} and unchanged-row "
        f"stratum mean {dense_sum['mean_gap_unchanged_stratum']['mean']}. "
        f"s2 reuse: mean actual-vs-fresh-full relL2="
        f"{s2_sum['mean_actual_full_rel_l2_reuse']['mean']}, mean stale-merge relL2="
        f"{s2_sum['mean_stale_merge_rel_l2_reuse']['mean']}. "
        f"Cache-label max relL2 dense/s2="
        f"{'n/a' if dense_sum['max_cache_label_rel_l2'] is None else dense_sum['max_cache_label_rel_l2']}/"
        f"{s2_sum['max_cache_label_rel_l2']}. "
        f"This run completed the fixed 16-prompt mechanism matrix with preflight and hard stops; "
        f"it is descriptive mechanism evidence on a small frozen subset, with no efficiency, "
        f"exactness, quality or heldout claim."
    )
    body = {
        "schema_version": 1, "run_id": RUN_ID, "status": status,
        "summary": summary_text,
        "executed_commands": commands,
        "metrics": metrics,
        "artifacts": [
            "preflight.json", "dense/generate-result.json", "s2/generate-result.json",
            "exports/mechanism-per-prompt.json", "exports/mechanism-summary.json",
            "exports/bounded-small-trace.json", "exports/resource-ledger.json",
            "exports/commands-and-provenance.json", "exports/mechanism-figure-data.csv",
        ],
        "deviations": [
            "matplotlib is not installed; exact CSV plotting data is provided instead of PNG/PDF "
            "(task explicitly permits this; no environment reinstall).",
            "Formal runs were executed as two sequential batch-1 processes (dense then s2) to keep "
            "the frozen batch-1 control and preserve per-condition evidence.",
            "Development probes were marked --allow-dirty and are not accepted scientific evidence; "
            "their conservative wall time is charged in the stage ledger.",
        ] + reasons,
        "failures": [
            "Development probe: KeyError('kv_heads') in initial hook registration; fixed before "
            "accepted preflight, raw output overwritten at the same dev path.",
            "Development probe: missing ninja executable in PATH for flashinfer JIT; fixed before "
            "accepted preflight, raw output overwritten at the same dev path.",
            "Development probe: initial paged-scatter negative-slot assertion targeted a written slot; "
            "assertion fixed, scatter implementation unchanged.",
        ],
        "resource_usage": {
            "wall_seconds": round(stage_wall, 3),
            "gpu_hours": round(stage_wall / 3600.0, 6),
            "formal_wall_seconds": round(formal_wall, 3),
            "accepted_preflight_wall_seconds": round(float(preflight_wall), 3),
            "development_probe_wall_seconds_charged": round(float(dev_wall), 3),
            "peak_allocated_mib_max": ledger["peak_allocated_mib_max"],
            "min_free_mib_sampled_min": ledger["min_free_mib_sampled_min"],
            "campaign_gpu_hours_before": cfg["budget"]["campaign_gpu_hours_before"],
            "campaign_gpu_hours_after_conservative": ledger["campaign_gpu_hours_after_conservative"],
        },
        "next_recommendation": (
            "GPT should review the paired gap, heterogeneity and s2 residual before authorizing any "
            "efficiency/quality stage. If the mechanism claim is retained, the next gate should use the "
            "frozen mechanism code to design the efficiency workload; no speed or quality conclusion "
            "follows from this run."
        ),
    }
    _atomic_json(out_dir / "report-body.json", body)
    print(json.dumps({"status": status, "mean_gap": primary["mean"],
                      "ci": [primary["ci_low"], primary["ci_high"]],
                      "out_dir": str(out_dir)}))
    return 0 if status == "SUCCEEDED" else 1


def _read_optional(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _validation_block(res):
    return {
        "status": res.get("status"),
        "execution_commit": res.get("execution_commit"),
        "finite_parameters": res.get("finite_parameters"),
        "logits_audit": res.get("logits_audit"),
        "engine_exit": res.get("engine_exit"),
        "capture_errors": res.get("capture_errors"),
        "n_prompts": len(res.get("prompts", [])),
        "tokens": [p.get("output", {}).get("new_tokens") for p in res.get("prompts", [])],
    }


# ---------------------------------------------------------------------------
# finalize report with version control
# ---------------------------------------------------------------------------
def cmd_finalize(args):
    cfg = load_config(args.config)
    run_dir = Path(args.run_dir).resolve()
    out_dir = run_dir / "exports"
    body = json.loads((out_dir / "report-body.json").read_text())
    head = _git_head()
    log = subprocess.check_output(
        ["git", "-C", str(ROOT), "log", "--format=%H%x09%s", "-40"], text=True)
    commits = []
    for line in log.strip().splitlines():
        sha, subj = line.split("\t", 1)
        if subj.startswith(RUN_ID) or subj.startswith("flashblock-20261023-prep-r3"):
            commits.append({"sha": sha, "subject": subj})
    run_commits = _read_optional(out_dir / "run-commits.json") or {}
    vc = {
        "repository": "Flutter-Misdreavus/flashblock-reproduction",
        "visibility": "private",
        "branch": "exp/flashblock-20261023",
        "run_commits": run_commits.get("run_commits", []),
        "head_commit": head,
        "commits": commits,
        "push_status": "pending",
        "push_error": "relay mode: executor commits locally only; GPT pulls the private repository objects and verifies/pushes",
        "push_mode": "relay",
    }
    body["version_control"] = vc
    body["run_commits"] = run_commits
    _atomic_json(run_dir / "report.json", body)
    print(json.dumps({"status": body.get("status"), "head_commit": head,
                      "report": str(run_dir / "report.json")}))
    return 0 if body.get("status") == "SUCCEEDED" else 1


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("unit")
    u.add_argument("--config", default=str(CONFIG_PATH))
    u.add_argument("--out", required=True)
    u.set_defaults(fn=cmd_unit)

    g = sub.add_parser("generate")
    g.add_argument("--config", default=str(CONFIG_PATH))
    g.add_argument("--run-dir", required=True)
    g.add_argument("--condition", required=True)
    g.add_argument("--ratio", type=float, required=True)
    g.add_argument("--hooks", type=int, default=1)
    g.add_argument("--indices", nargs="+", type=int, required=True)
    g.add_argument("--max-tokens", type=int, default=512)
    g.add_argument("--out", required=True)
    g.add_argument("--raw-dir", default=None)
    g.add_argument("--allow-dirty", action="store_true")
    g.set_defaults(fn=cmd_generate)

    p = sub.add_parser("preflight")
    p.add_argument("--config", default=str(CONFIG_PATH))
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--attempt", type=int, default=1)
    p.set_defaults(fn=cmd_preflight)

    s = sub.add_parser("summarize")
    s.add_argument("--config", default=str(CONFIG_PATH))
    s.add_argument("--run-dir", required=True)
    s.add_argument("--dense", required=True)
    s.add_argument("--s2", required=True)
    s.set_defaults(fn=cmd_summarize)

    f = sub.add_parser("finalize")
    f.add_argument("--config", default=str(CONFIG_PATH))
    f.add_argument("--run-dir", required=True)
    f.set_defaults(fn=cmd_finalize)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
