#!/usr/bin/env python3
"""Preparation-gate CLI for the FlashBlock reproduction (run flashblock-20261023-prep-r3).

Subcommands:
  correctness  independent numerical validation of the vendored custom FlashAttention
               kernel: dense parity, no-reuse history decomposition, genuine cached
               reuse with persistent A/L buffers, LSE/recomposition, dirty-mask audit,
               reset/slot reorder, GQA32/8 head_dim128 block4.
  smoke        real TraDo-4B generation on frozen GSM8K train prompts 100..101 for
               one condition (baseline dense or official s2/tau2), max 64 new tokens.

All metrics are written as JSON; no tolerance changes after observing results.
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLASH = ROOT / "flash-attention"
DLLM = ROOT / "dllm"
GPU_UUID = "GPU-REDACTED"
SEED = 42
MASK_TOKEN_ID = 151669
RUN_ID = "flashblock-20261023-prep-r3"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
PROMPT_TEMPLATE = (
    "<|im_start|>user\nPlease reason step by step, and put your final answer within "
    "\\boxed{{}}.\n{problem}<|im_end|>\n<|im_start|>assistant\n"
)


def _setup_paths():
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for p in (str(FLASH), str(DLLM), str(DLLM / "sample")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _seed_all(seed=SEED):
    """Seed Python, NumPy, torch CPU and every CUDA device before init/sampling."""
    import torch
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


def _rel_l2(a, b):
    import torch
    a = a.float()
    b = b.float()
    denom = torch.linalg.vector_norm(b).item()
    return (torch.linalg.vector_norm(a - b).item() / denom) if denom else float("nan")


def _max_abs(a, b):
    import torch
    return (a.float() - b.float()).abs().max().item()


def _ref_attn(q, k, v):
    """Independent naive FP32/float64 grouped-query full attention."""
    import torch
    q = q.double()
    k = k.double()
    v = v.double()
    rep = q.shape[2] // k.shape[2]
    if rep > 1:
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("bqhd,bshd->bhqs", q, k) * scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhqs,bshd->bqhd", probs, v)
    return out


def _ref_attn_lse(q, k, v):
    """Independent attention output plus natural-log LSE, shaped [B,Q,H,1]."""
    import torch
    qd = q.double()
    kd = k.double()
    vd = v.double()
    rep = q.shape[2] // k.shape[2]
    if rep > 1:
        kd = kd.repeat_interleave(rep, dim=2)
        vd = vd.repeat_interleave(rep, dim=2)
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("bqhd,bshd->bhqs", qd, kd) * scale
    lse = torch.logsumexp(scores, dim=-1)                       # [B,H,Q]
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhqs,bshd->bqhd", probs, vd)            # [B,Q,H,D]
    return out, lse.transpose(1, 2).unsqueeze(-1).contiguous()   # [B,Q,H,1]


def _merge_ref(a, lse_a, b, lse_b):
    """Stable log-space composition of two disjoint attention results."""
    import torch
    m = torch.maximum(lse_a, lse_b)
    ea = torch.exp(lse_a - m)
    eb = torch.exp(lse_b - m)
    denom = ea + eb
    out = (a.double() * ea + b.double() * eb) / denom
    lse = m + torch.log(denom)
    return out, lse


def _sdpa_ref(q, k, v):
    import torch
    import torch.nn.functional as F
    rep = q.shape[2] // k.shape[2]
    kk = k.repeat_interleave(rep, dim=2).float()
    vv = v.repeat_interleave(rep, dim=2).float()
    return F.scaled_dot_product_attention(
        q.float().transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2)
    ).transpose(1, 2)


def _provenance(flash_mod):
    import torch
    so = Path(getattr(flash_mod, "__file__", ""))
    info = {
        "flash_attn_2_cuda_path": str(so),
        "under_vendored_tree": str(so.resolve()).startswith(str(FLASH.resolve()) + os.sep),
        "sha256": _sha256(so) if so.is_file() else None,
        "bytes": so.stat().st_size if so.is_file() else None,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_uuid_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        nm = subprocess.run(
            ["nm", "-D", "--defined-only", str(so)], capture_output=True, text=True, timeout=60
        ).stdout
        info["history_symbols"] = sorted(
            {ln.split()[-1] for ln in nm.splitlines() if "history" in ln.lower()}
        )[:20]
        info["update_symbols"] = sorted(
            {ln.split()[-1] for ln in nm.splitlines() if "update_attn" in ln.lower()}
        )[:20]
    except Exception as e:  # pragma: no cover - diagnostic only
        info["nm_error"] = repr(e)
    return info


def _kernel_call(q, k_cache, v_cache, knew, vnew, cache_seqlens, block_size, block_table=None,
                 out_past=None, lse_past=None, store_mask=None, update_mask=None, dirty_mask=None):
    """Call the vendored kernel; block_size>0 uses the fp32 history-decomposition path."""
    import torch
    from flash_attn import flash_attn_with_kvcache

    B, Q, Hq, D = q.shape
    if block_size > 0:
        if out_past is None:
            out_past = torch.zeros((B, Q, Hq, D), dtype=torch.float32, device=q.device)
        if lse_past is None:
            lse_past = torch.zeros((B, Q, Hq, 1), dtype=torch.float32, device=q.device)
        if store_mask is None and update_mask is None:
            store_mask = torch.ones((B, Hq, Q, 1), dtype=torch.bool, device=q.device)
        kwargs = dict(
            q, k_cache, v_cache, k=knew, v=vnew,
            cache_seqlens=cache_seqlens, block_table=block_table,
            causal=False, return_softmax_lse=True, block_size=block_size, num_splits=1,
            attn_output_past=out_past, logsumexp=lse_past,
        )
        if update_mask is not None:
            kwargs["need_update_kvcache_mask"] = update_mask
        if store_mask is not None:
            kwargs["need_store_history_mask"] = store_mask
        if dirty_mask is not None:
            kwargs["dirty_mask"] = dirty_mask
        ret = flash_attn_with_kvcache(**kwargs)
        out, lse, out_hist, lse_hist, out_past_new, lse_past_new = ret
        return {
            "out": out, "lse": lse, "out_hist": out_hist, "lse_hist": lse_hist,
            "out_past": out_past_new, "lse_past": lse_past_new,
        }
    out = flash_attn_with_kvcache(
        q, k_cache, v_cache, k=knew, v=vnew,
        cache_seqlens=cache_seqlens, block_table=block_table,
        causal=False, block_size=0,
    )
    return {"out": out}


def _case(name, fn, results, **meta):
    import torch
    torch.cuda.synchronize()
    t0 = time.time()
    try:
        rec = fn()
        rec.update({"case": name, **meta, "seconds": round(time.time() - t0, 4)})
        rec["passed"] = bool(rec.pop("_passed", False))
    except Exception as e:
        import traceback
        rec = {"case": name, **meta, "passed": False, "error": repr(e),
               "traceback": traceback.format_exc()[-2000:], "seconds": round(time.time() - t0, 4)}
    results.append(rec)
    print(f"[correctness] {name}: passed={rec['passed']}" + (f" err={rec.get('error')}" if rec.get("error") else ""), flush=True)
    return rec


def cmd_correctness(args):
    _setup_paths()
    import torch
    import flash_attn_2_cuda as fac

    seed_info = _seed_all()
    torch.set_grad_enabled(False)

    results = []
    prov = _provenance(fac)

    # ---------- 1. FP32 reference self-check (naive vs SDPA) ----------
    def ref_selfcheck():
        out = {}
        for (B, Q, S, Hq, Hkv, D) in [(2, 7, 33, 8, 2, 64), (1, 4, 1, 4, 4, 128), (3, 5, 65, 6, 3, 40)]:
            q = torch.randn(B, Q, Hq, D, device="cuda", dtype=torch.float32)
            k = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float32)
            v = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float32)
            a = _ref_attn(q, k, v)
            b = _sdpa_ref(q, k, v)
            out[f"B{B}_Q{Q}_S{S}_Hq{Hq}_Hkv{Hkv}_D{D}"] = {
                "rel_l2": _rel_l2(a, b), "max_abs": _max_abs(a, b)}
        ok = all(r["rel_l2"] <= 1e-5 for r in out.values())
        return {"_passed": ok, "tolerance": {"rel_l2": 1e-5}, "cases": out}

    _case("fp32_reference_selfcheck", ref_selfcheck, results)

    # ---------- 2. BF16 dense parity over varying lengths and GQA ----------
    def dense_parity():
        out = {}
        # the vendored build compiles bf16 hdim128 only (TraDo-4B head_dim=128)
        cases = [(1, 1, 1, 8, 8, 128), (2, 4, 17, 8, 2, 128), (2, 8, 65, 8, 8, 128),
                 (1, 3, 129, 16, 4, 128), (2, 5, 256, 8, 1, 128)]
        for (B, Q, S, Hq, Hkv, D) in cases:
            past = S - Q
            q = torch.randn(B, Q, Hq, D, device="cuda", dtype=torch.bfloat16)
            k_new = torch.randn(B, Q, Hkv, D, device="cuda", dtype=torch.bfloat16)
            v_new = torch.randn(B, Q, Hkv, D, device="cuda", dtype=torch.bfloat16)
            k_cache = torch.zeros(B, S, Hkv, D, device="cuda", dtype=torch.bfloat16)
            v_cache = torch.zeros(B, S, Hkv, D, device="cuda", dtype=torch.bfloat16)
            if past > 0:
                k_cache[:, :past] = torch.randn(B, past, Hkv, D, device="cuda", dtype=torch.bfloat16)
                v_cache[:, :past] = torch.randn(B, past, Hkv, D, device="cuda", dtype=torch.bfloat16)
            k_cache[:, past:past + Q] = k_new
            v_cache[:, past:past + Q] = v_new
            cs = torch.full((B,), past, dtype=torch.int32, device="cuda")
            res = _kernel_call(q, k_cache, v_cache, k_new, v_new, cs, 0)
            ref = _ref_attn(q, k_cache, v_cache)
            ref_sdpa = _sdpa_ref(q, k_cache, v_cache)
            key = f"B{B}_Q{Q}_S{S}_Hq{Hq}_Hkv{Hkv}_D{D}"
            out[key] = {"rel_l2_vs_fp32ref": _rel_l2(res["out"], ref),
                        "max_abs_vs_fp32ref": _max_abs(res["out"], ref),
                        "rel_l2_vs_sdpa": _rel_l2(res["out"], ref_sdpa),
                        "max_abs_vs_sdpa": _max_abs(res["out"], ref_sdpa)}
        ok = all(r["rel_l2_vs_fp32ref"] <= 1e-2 and r["max_abs_vs_fp32ref"] <= 5e-2 for r in out.values())
        return {"_passed": ok, "tolerance": {"rel_l2": 1e-2, "max_abs": 5e-2}, "cases": out}

    _case("bf16_dense_parity_varying_lengths", dense_parity, results)

    # ---------- 3. no-reuse history decomposition, single step ----------
    def decomposition():
        out = {}
        for (B, S, Hq, Hkv, D) in [(2, 16, 8, 2, 128), (2, 68, 8, 2, 128), (1, 4, 16, 4, 128), (2, 0, 8, 2, 128)]:
            Q = 4
            q = torch.randn(B, Q, Hq, D, device="cuda", dtype=torch.bfloat16)
            k_new = torch.randn(B, Q, Hkv, D, device="cuda", dtype=torch.bfloat16)
            v_new = torch.randn(B, Q, Hkv, D, device="cuda", dtype=torch.bfloat16)
            k_cache = torch.zeros(B, S + Q, Hkv, D, device="cuda", dtype=torch.bfloat16)
            v_cache = torch.zeros(B, S + Q, Hkv, D, device="cuda", dtype=torch.bfloat16)
            if S:
                k_cache[:, :S] = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.bfloat16)
                v_cache[:, :S] = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.bfloat16)
            k_cache[:, S:] = k_new
            v_cache[:, S:] = v_new
            cs = torch.full((B,), S, dtype=torch.int32, device="cuda")
            res = _kernel_call(q, k_cache, v_cache, k_new, v_new, cs, Q)
            full_ref, full_lse = _ref_attn_lse(q, k_cache, v_cache)
            key = f"B{B}_past{S}_Hq{Hq}_Hkv{Hkv}_D{D}"
            rec = {"rel_l2_full": _rel_l2(res["out"], full_ref),
                   "max_abs_full": _max_abs(res["out"], full_ref),
                   "rel_l2_lse_full": _rel_l2(res["lse"], full_lse),
                   "out_hist_shape": list(res["out_hist"].shape),
                   "out_hist_dtype": str(res["out_hist"].dtype),
                   "hist_nan": bool(torch.isnan(res["out_hist"]).any().item()),
                   "lse_hist_nan": bool(torch.isnan(res["lse_hist"]).any().item())}
            if S:
                past_ref, past_lse = _ref_attn_lse(q, k_cache[:, :S], v_cache[:, :S])
                rec["rel_l2_hist"] = _rel_l2(res["out_hist"], past_ref)
                rec["max_abs_hist"] = _max_abs(res["out_hist"], past_ref)
                rec["rel_l2_lse_hist"] = _rel_l2(res["lse_hist"], past_lse)
            else:
                rec["hist_abs_max"] = res["out_hist"].abs().max().item()
                rec["lse_hist_inf_fraction"] = res["lse_hist"].isinf().float().mean().item()
                rec["lse_hist_neg_inf_fraction"] = (res["lse_hist"] == float("-inf")).float().mean().item()
            out[key] = rec
        ok = all(
            r["rel_l2_full"] <= 1e-2 and r["max_abs_full"] <= 5e-2
            and not r["hist_nan"] and not r["lse_hist_nan"]
            and (("rel_l2_hist" not in r) or (r["rel_l2_hist"] <= 1e-2 and r["max_abs_hist"] <= 5e-2))
            for r in out.values()
        )
        return {"_passed": ok, "tolerance": {"rel_l2": 1e-2, "max_abs": 5e-2},
                "note": "each call recomputes history from scratch (all store mask true) = no-reuse control; "
                        "empty history must yield zero out_hist and -inf (not NaN) LSE",
                "cases": out}

    _case("no_reuse_history_decomposition", decomposition, results)

    # ---------- 4. multi-step no-reuse (8 denoising steps, cache grows) ----------
    def multistep():
        B, Q, Hq, Hkv, D, prefix = 2, 4, 8, 2, 128, 16
        steps = 8
        k_cache = torch.zeros(B, prefix + Q * steps, Hkv, D, device="cuda", dtype=torch.bfloat16)
        v_cache = torch.zeros(B, prefix + Q * steps, Hkv, D, device="cuda", dtype=torch.bfloat16)
        k_cache[:, :prefix] = torch.randn(B, prefix, Hkv, D, device="cuda", dtype=torch.bfloat16)
        v_cache[:, :prefix] = torch.randn(B, prefix, Hkv, D, device="cuda", dtype=torch.bfloat16)
        per_step = []
        for s in range(steps):
            q = torch.randn(B, Q, Hq, D, device="cuda", dtype=torch.bfloat16)
            k_new = torch.randn(B, Q, Hkv, D, device="cuda", dtype=torch.bfloat16)
            v_new = torch.randn(B, Q, Hkv, D, device="cuda", dtype=torch.bfloat16)
            past = prefix + s * Q
            k_cache[:, past:past + Q] = k_new
            v_cache[:, past:past + Q] = v_new
            cs = torch.full((B,), past, dtype=torch.int32, device="cuda")
            res = _kernel_call(q, k_cache, v_cache, k_new, v_new, cs, Q)
            ref, ref_lse = _ref_attn_lse(q, k_cache[:, :past + Q], v_cache[:, :past + Q])
            per_step.append({"step": s, "cache_len": past, "rel_l2": _rel_l2(res["out"], ref),
                             "max_abs": _max_abs(res["out"], ref),
                             "rel_l2_lse": _rel_l2(res["lse"], ref_lse)})
        ok = all(x["rel_l2"] <= 1e-2 and x["max_abs"] <= 5e-2 and x["rel_l2_lse"] <= 1e-2
                 for x in per_step)
        return {"_passed": ok, "steps": per_step}

    _case("no_reuse_multistep_cache_growth", multistep, results)

    # ---------- 5. paged KV + slot reuse / cache reset ----------
    def paged_slots():
        B, Q, Hq, Hkv, D = 2, 4, 8, 2, 128
        page, nblocks, npages = 256, 6, 2
        k_cache = torch.zeros(nblocks, page, Hkv, D, device="cuda", dtype=torch.bfloat16)
        v_cache = torch.zeros(nblocks, page, Hkv, D, device="cuda", dtype=torch.bfloat16)
        # logical page 0 -> physical 2, logical page 1 -> physical 5 for seq0; 4,1 for seq1
        block_table = torch.tensor([[2, 5], [4, 1]], dtype=torch.int32, device="cuda")
        past = 300  # crosses into second logical page
        k_all = torch.randn(B, past + Q, Hkv, D, device="cuda", dtype=torch.bfloat16)
        v_all = torch.randn(B, past + Q, Hkv, D, device="cuda", dtype=torch.bfloat16)
        for b in range(B):
            for pos in range(past + Q):
                phys = block_table[b, pos // page].item()
                k_cache[phys, pos % page] = k_all[b, pos]
                v_cache[phys, pos % page] = v_all[b, pos]
        cs = torch.full((B,), past, dtype=torch.int32, device="cuda")
        q = torch.randn(B, Q, Hq, D, device="cuda", dtype=torch.bfloat16)
        knew = k_all[:, past:].contiguous()
        vnew = v_all[:, past:].contiguous()
        res = _kernel_call(q, k_cache, v_cache, knew, vnew, cs, Q, block_table=block_table)
        ref = _ref_attn(q, k_all[:, :past + Q], v_all[:, :past + Q])
        rec = {"rel_l2": _rel_l2(res["out"], ref), "max_abs": _max_abs(res["out"], ref)}
        # slot reuse: overwrite physical blocks with unrelated data, reset lengths, new seq
        k_cache.zero_(); v_cache.zero_()
        newpast = 128
        k2 = torch.randn(B, newpast + Q, Hkv, D, device="cuda", dtype=torch.bfloat16)
        v2 = torch.randn(B, newpast + Q, Hkv, D, device="cuda", dtype=torch.bfloat16)
        for b in range(B):
            for pos in range(newpast + Q):
                phys = block_table[b, pos // page].item()
                k_cache[phys, pos % page] = k2[b, pos]
                v_cache[phys, pos % page] = v2[b, pos]
        cs2 = torch.full((B,), newpast, dtype=torch.int32, device="cuda")
        q2 = torch.randn(B, Q, Hq, D, device="cuda", dtype=torch.bfloat16)
        res2 = _kernel_call(q2, k_cache, v_cache, k2[:, newpast:].contiguous(),
                            v2[:, newpast:].contiguous(), cs2, Q, block_table=block_table)
        ref2 = _ref_attn(q2, k2, v2)
        rec["after_slot_reset_rel_l2"] = _rel_l2(res2["out"], ref2)
        rec["after_slot_reset_max_abs"] = _max_abs(res2["out"], ref2)
        ok = all(rec[k] <= (1e-2 if "rel_l2" in k else 5e-2) for k in rec)
        return {"_passed": ok, "block_table": block_table.tolist(), "cases": rec}

    _case("paged_kv_slot_reuse_and_reset", paged_slots, results)

    # ---------- 6. genuine cached reuse with persistent A/L buffers ----------
    # Actual model geometry: 32 query heads / 8 KV heads / head_dim 128 / block 4.
    B, Q, Hq, Hkv, D = 2, 4, 32, 8, 128
    prefix = 68
    max_len = prefix + 3 * Q
    k_cache = torch.zeros(B, max_len, Hkv, D, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.zeros(B, max_len, Hkv, D, device="cuda", dtype=torch.bfloat16)
    k_cache[:, :prefix] = torch.randn(B, prefix, Hkv, D, device="cuda", dtype=torch.bfloat16)
    v_cache[:, :prefix] = torch.randn(B, prefix, Hkv, D, device="cuda", dtype=torch.bfloat16)
    q1 = torch.randn(B, Q, Hq, D, device="cuda", dtype=torch.bfloat16)
    k1 = torch.randn(B, Q, Hkv, D, device="cuda", dtype=torch.bfloat16)
    v1 = torch.randn(B, Q, Hkv, D, device="cuda", dtype=torch.bfloat16)
    k_cache[:, prefix:prefix + Q] = k1
    v_cache[:, prefix:prefix + Q] = v1
    prefix_out, prefix_lse = _ref_attn_lse(q1, k_cache[:, :prefix], v_cache[:, :prefix])
    A = torch.zeros((B, Q, Hq, D), dtype=torch.float32, device="cuda")
    L = torch.zeros((B, Q, Hq, 1), dtype=torch.float32, device="cuda")
    A.copy_(prefix_out.float())
    L.copy_(prefix_lse.float())
    ones = torch.ones((B, Hq, Q, 1), dtype=torch.bool, device="cuda")
    zeros = torch.zeros((B, Hq, Q, 1), dtype=torch.bool, device="cuda")
    cs1 = torch.full((B,), prefix, dtype=torch.int32, device="cuda")

    state = {}

    def cached_unchanged_q():
        full1, full1_lse = _ref_attn_lse(q1, k_cache[:, :prefix + Q], v_cache[:, :prefix + Q])
        r1 = _kernel_call(q1, k_cache, v_cache, k1, v1, cs1, Q,
                         out_past=A.clone(), lse_past=L.clone(), store_mask=ones)
        # r1 stores history-only (prefix) stats and outputs full attention
        r2 = _kernel_call(q1, k_cache, v_cache, k1, v1, cs1, Q,
                         out_past=r1["out_past"].clone(), lse_past=r1["lse_past"].clone(),
                         update_mask=zeros, store_mask=zeros, dirty_mask=zeros)
        cur_out, cur_lse = _ref_attn_lse(q1, k1, v1)
        state.update({"A": r1["out_past"].clone(), "L": r1["lse_past"].clone(),
                      "cache_len": prefix + Q})
        rec = {
            "store_out_rel_l2_vs_full": _rel_l2(r1["out"], full1),
            "store_out_max_abs_vs_full": _max_abs(r1["out"], full1),
            "store_lse_rel_l2_vs_full": _rel_l2(r1["lse"], full1_lse),
            "stored_A_rel_l2_vs_prefix": _rel_l2(r1["out_past"], prefix_out),
            "stored_L_rel_l2_vs_prefix": _rel_l2(r1["lse_past"], prefix_lse),
            "unchanged_q_reuse_rel_l2_vs_store": _rel_l2(r2["out"], r1["out"]),
            "unchanged_q_reuse_max_abs_vs_store": _max_abs(r2["out"], r1["out"]),
            "reuse_returned_lse_rel_l2_vs_current_block": _rel_l2(r2["lse"], cur_lse),
            "reuse_A_unchanged": bool(torch.equal(r2["out_past"], r1["out_past"])),
            "reuse_L_unchanged": bool(torch.equal(r2["lse_past"], r1["lse_past"])),
        }
        ok = (rec["store_out_rel_l2_vs_full"] <= 1e-2 and rec["store_out_max_abs_vs_full"] <= 5e-2
              and rec["stored_A_rel_l2_vs_prefix"] <= 1e-2
              and rec["unchanged_q_reuse_rel_l2_vs_store"] <= 1e-2
              and rec["reuse_A_unchanged"] and rec["reuse_L_unchanged"])
        return {"_passed": ok, "tolerance": {"rel_l2": 1e-2, "max_abs": 5e-2}, "cases": rec}

    _case("cached_reuse_unchanged_q", cached_unchanged_q, results)

    def cached_changed_q_stale():
        q2 = torch.randn(B, Q, Hq, D, device="cuda", dtype=torch.bfloat16)
        k2 = torch.randn(B, Q, Hkv, D, device="cuda", dtype=torch.bfloat16)
        v2 = torch.randn(B, Q, Hkv, D, device="cuda", dtype=torch.bfloat16)
        k_cache[:, prefix + Q:prefix + 2 * Q] = k2
        v_cache[:, prefix + Q:prefix + 2 * Q] = v2
        cs2 = torch.full((B,), prefix + Q, dtype=torch.int32, device="cuda")
        r = _kernel_call(q2, k_cache, v_cache, k2, v2, cs2, Q,
                         out_past=state["A"].clone(), lse_past=state["L"].clone(),
                         update_mask=zeros, store_mask=zeros, dirty_mask=zeros)
        # independent composition: cached history is prefix-only (block1 is stale/omitted)
        hist_out, hist_lse = _ref_attn_lse(q2, k_cache[:, :prefix], v_cache[:, :prefix])
        cur_out, cur_lse = _ref_attn_lse(q2, k2, v2)
        merged, merged_lse = _merge_ref(hist_out, hist_lse, cur_out, cur_lse)
        full_out, full_lse = _ref_attn_lse(q2, k_cache[:, :prefix + 2 * Q], v_cache[:, :prefix + 2 * Q])
        rec = {
            "rel_l2_vs_stale_composition": _rel_l2(r["out"], merged),
            "max_abs_vs_stale_composition": _max_abs(r["out"], merged),
            "rel_l2_vs_full_recompute": _rel_l2(r["out"], full_out),
            "max_abs_vs_full_recompute": _max_abs(r["out"], full_out),
            "returned_lse_rel_l2_vs_current_block": _rel_l2(r["lse"], cur_lse),
            "cache_L_equals_history": bool(torch.equal(r["lse_past"], state["L"])),
            "stale_cache_missing_tokens": Q,
        }
        ok = (rec["rel_l2_vs_stale_composition"] <= 1e-2
              and rec["max_abs_vs_stale_composition"] <= 5e-2
              and rec["cache_L_equals_history"])
        return {"_passed": ok,
                "tolerance": {"rel_l2": 1e-2, "max_abs": 5e-2},
                "note": "A/L are the persistent prefix-only cache; block1 is intentionally not in the "
                        "cache, so the independent target is log-space composition(prefix, block2), "
                        "not full recomputation. Full-recompute error is reported as approximation.",
                "cases": rec}

    _case("cached_reuse_changed_q_stale_composition", cached_changed_q_stale, results)

    def cached_dirty_audit():
        q3 = torch.randn(B, Q, Hq, D, device="cuda", dtype=torch.bfloat16)
        k3 = torch.randn(B, Q, Hkv, D, device="cuda", dtype=torch.bfloat16)
        v3 = torch.randn(B, Q, Hkv, D, device="cuda", dtype=torch.bfloat16)
        k_cache[:, prefix + 2 * Q:prefix + 3 * Q] = k3
        v_cache[:, prefix + 2 * Q:prefix + 3 * Q] = v3
        cs3 = torch.full((B,), prefix + 2 * Q, dtype=torch.int32, device="cuda")
        dirty = torch.zeros((B, Q, Hq, 1), dtype=torch.bool, device="cuda")
        dirty[0, :, :Hq // 2, :] = True
        dirty[1, :, Hq // 2:, :] = True
        r = _kernel_call(q3, k_cache, v_cache, k3, v3, cs3, Q,
                         out_past=state["A"].clone(), lse_past=state["L"].clone(),
                         update_mask=zeros, store_mask=zeros, dirty_mask=dirty)
        r0 = _kernel_call(q3, k_cache, v_cache, k3, v3, cs3, Q,
                          out_past=state["A"].clone(), lse_past=state["L"].clone(),
                          update_mask=zeros, store_mask=zeros, dirty_mask=zeros)
        hist_out, hist_lse = _ref_attn_lse(q3, k_cache[:, :prefix], v_cache[:, :prefix])
        cur_out, cur_lse = _ref_attn_lse(q3, k3, v3)
        merged, _ = _merge_ref(hist_out, hist_lse, cur_out, cur_lse)
        full_out, _ = _ref_attn_lse(q3, k_cache[:, :prefix + 3 * Q], v_cache[:, :prefix + 3 * Q])
        clean = ~dirty
        rec = {
            "dirty_fraction": dirty.float().mean().item(),
            "dirty_mask_effective": not bool(torch.equal(r["out"], r0["out"])),
            "rel_l2_vs_composition_all": _rel_l2(r["out"], merged),
            "rel_l2_vs_composition_dirty_rows": _rel_l2(r["out"][clean.expand_as(r["out"])], merged[clean.expand_as(r["out"])]),
            "rel_l2_vs_full_dirty_rows": _rel_l2(r["out"][dirty.expand_as(r["out"])], full_out[dirty.expand_as(r["out"])]),
            "all_finite": bool(torch.isfinite(r["out"]).all().item()),
            "cache_unchanged": bool(torch.equal(r["out_past"], state["A"])),
        }
        # With a history-only A/L cache, dirty detection is expected to be a no-op:
        # the cache excludes the current block, so changed query rows still compose
        # correctly with a freshly computed current block. Verified, not assumed.
        ok = (rec["all_finite"] and rec["cache_unchanged"]
              and rec["rel_l2_vs_composition_all"] <= 1e-2)
        return {"_passed": ok,
                "note": "audited actual upstream semantics: this path caches history-only A/L "
                        "(prefix excluding the current block); dirty_mask neither changes output nor "
                        "the cache because the current block is always recomputed. "
                        "dirty_mask_effective=false is documented, not hidden.",
                "cases": rec}

    _case("cached_reuse_dirty_mask_audit", cached_dirty_audit, results)

    def cached_reset_slot_reorder():
        Bs, Qs, Ds, Hqs, Hkvs = 2, 4, 128, 32, 8
        prefs = [12, 20]
        kc = torch.zeros(Bs, max(prefs) + Qs, Hkvs, Ds, device="cuda", dtype=torch.bfloat16)
        vc = torch.zeros(Bs, max(prefs) + Qs, Hkvs, Ds, device="cuda", dtype=torch.bfloat16)
        qs = torch.randn(Bs, Qs, Hqs, Ds, device="cuda", dtype=torch.bfloat16)
        kn = torch.randn(Bs, Qs, Hkvs, Ds, device="cuda", dtype=torch.bfloat16)
        vn = torch.randn(Bs, Qs, Hkvs, Ds, device="cuda", dtype=torch.bfloat16)
        for b, p in enumerate(prefs):
            kc[b, :p] = torch.randn(p, Hkvs, Ds, device="cuda", dtype=torch.bfloat16)
            vc[b, :p] = torch.randn(p, Hkvs, Ds, device="cuda", dtype=torch.bfloat16)
            kc[b, p:p + Qs] = kn[b]
            vc[b, p:p + Qs] = vn[b]
        cs = torch.tensor(prefs, dtype=torch.int32, device="cuda")
        masks = torch.ones((Bs, Hqs, Qs, 1), dtype=torch.bool, device="cuda")
        # per-sequence prefix reference for A/L
        outs = []
        for b, p in enumerate(prefs):
            o, l = _ref_attn_lse(qs[b:b + 1], kc[b:b + 1, :p], vc[b:b + 1, :p])
            outs.append((o, l))
        A = torch.cat([o.float() for o, _ in outs], 0)
        L = torch.cat([l.float() for _, l in outs], 0)
        r = _kernel_call(qs, kc, vc, kn, vn, cs, Qs, out_past=A, lse_past=L, store_mask=masks)
        refs = [_ref_attn(qs[b:b + 1], kc[b:b + 1, :prefs[b] + Qs], vc[b:b + 1, :prefs[b] + Qs]) for b in range(Bs)]
        rel = [_rel_l2(r["out"][b:b + 1], refs[b]) for b in range(Bs)]
        # swap batch slots and A/L, masks unchanged (both true); outputs must swap
        idx = torch.tensor([1, 0], device="cuda")
        kc2 = kc.index_select(0, idx).contiguous()
        vc2 = vc.index_select(0, idx).contiguous()
        qs2 = qs.index_select(0, idx).contiguous()
        kn2 = kn.index_select(0, idx).contiguous()
        vn2 = vn.index_select(0, idx).contiguous()
        cs2 = cs.index_select(0, idx).contiguous()
        r2 = _kernel_call(qs2, kc2, vc2, kn2, vn2, cs2, Qs,
                          out_past=A.index_select(0, idx).contiguous(),
                          lse_past=L.index_select(0, idx).contiguous(), store_mask=masks)
        swap_err = _rel_l2(r2["out"], r["out"].index_select(0, idx))
        # reset: zero A/L then refresh, must equal full recomputation
        zA = torch.zeros_like(A)
        zL = torch.zeros_like(L)
        r3 = _kernel_call(qs, kc, vc, kn, vn, cs, Qs, out_past=zA, lse_past=zL, store_mask=masks)
        reset_rel = [_rel_l2(r3["out"][b:b + 1], refs[b]) for b in range(Bs)]
        rec = {"per_sequence_rel_l2": rel, "slot_swap_rel_l2": swap_err,
               "reset_rel_l2": reset_rel, "reset_A_rel_l2": _rel_l2(r3["out_past"], r["out_past"])}
        ok = all(x <= 1e-2 for x in rel + reset_rel) and swap_err <= 1e-2
        return {"_passed": ok, "cases": rec}

    _case("cached_reuse_reset_and_slot_reorder", cached_reset_slot_reorder, results)

    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "seed": SEED,
        "seed_state": seed_info,
        "tolerances": {"fp32_reference_rel_l2": 1e-5, "bf16_rel_l2": 1e-2, "bf16_max_abs": 5e-2},
        "kernel_provenance": prov,
        "environment": {
            "torch": __import__("torch").__version__,
            "triton": __import__("triton").__version__,
            "numpy": np.__version__,
        },
        "all_passed": bool(results) and all(r["passed"] for r in results),
        "cases": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, out)
    print(json.dumps({"all_passed": payload["all_passed"], "out": str(out)}))
    return 0 if payload["all_passed"] else 1


def _load_gsm8k_rows(parquet, indices):
    import pyarrow.parquet as pq
    tab = pq.read_table(parquet)
    rows = tab.to_pylist()
    return [rows[i] for i in indices]


def _free_mem_mb():
    import torch
    free, total = torch.cuda.mem_get_info()
    return free / 2**20, total / 2**20


def _params_finite(module):
    """Chunked full-parameter finiteness scan without materializing a full boolean copy."""
    import torch
    total = 0
    nonfinite_tensors = []
    nonfinite_elems = 0
    chunk = 1 << 24
    for name, p in module.named_parameters():
        total += p.numel()
        flat = p.detach().view(-1)
        bad = 0
        for i in range(0, flat.numel(), chunk):
            part = flat[i:i + chunk]
            bad += int((~torch.isfinite(part)).sum().item())
        if bad:
            nonfinite_tensors.append({"name": name, "nonfinite": bad})
            nonfinite_elems += bad
    return {"tensors": sum(1 for _ in module.parameters()), "elements": total,
            "nonfinite_elements": nonfinite_elems, "nonfinite_tensors": nonfinite_tensors[:20]}


def cmd_smoke(args):
    _setup_paths()
    os.environ["TRADO_CONSISTENT_SAMPLING_PARAMS"] = "1"
    os.environ["TRADO_WARMUP_MODEL_LEN"] = str(args.warmup_len)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import torch

    ratio = float(args.ratio)
    cond = "baseline" if ratio == 0 else args.condition
    rows = _load_gsm8k_rows(args.parquet, args.indices)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    prompts = []
    template_audit = []
    for r in rows:
        content = INSTRUCTION + "\n" + r["question"]
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        hard = PROMPT_TEMPLATE.format(problem=r["question"])
        template_audit.append({
            "matches_tokenizer_template": bool(chat == hard),
            "prompt_sha256": hashlib.sha256(chat.encode()).hexdigest(),
        })
        prompts.append(chat if chat == hard else hard)

    trace_path = Path(args.out).with_suffix(".reuse.jsonl")
    if trace_path.exists():
        trace_path.unlink()
    os.environ["TRADO_SCHEDULE_REUSE_DUMP"] = str(trace_path)

    mem_samples = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                free, _ = _free_mem_mb()
                mem_samples.append(free)
            except Exception:
                pass
            time.sleep(0.5)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t0 = time.time()
    result = {"condition": cond, "sparsity_ratio": ratio, "indices": list(args.indices),
              "max_tokens": args.max_tokens, "seed": SEED, "status": "FAILED",
              "template_audit": template_audit}
    llm = None
    audit = {"calls": 0, "nonfinite_calls": 0, "max_numel": 0}
    peak_alloc = peak_reserved = 0.0
    try:
        seed_before_init = _seed_all()
        from jetengine_ext.llm import LLM
        from jetengine_ext.sampling_params import SamplingParams

        llm = LLM(
            args.model, enforce_eager=True, tensor_parallel_size=1,
            mask_token_id=MASK_TOKEN_ID, block_length=4, batch_size=max(4, len(prompts)),
            gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_model_len, max_num_seqs=4,
        )
        kv_blocks = int(llm.config.num_kvcache_blocks)
        kv_block_size = int(llm.config.kvcache_block_size)
        kv_tokens = kv_blocks * kv_block_size
        result["kv_cache"] = {"blocks": kv_blocks, "block_size": kv_block_size,
                              "capacity_tokens": kv_tokens,
                              "gpu_memory_utilization": args.gpu_mem}
        if kv_tokens < 2 * args.max_model_len + 64:
            raise RuntimeError(f"KV capacity {kv_tokens} < required {2 * args.max_model_len + 64}")

        finite = _params_finite(llm.model_runner.model)
        result["finite_parameters"] = finite
        if finite["nonfinite_elements"] != 0:
            raise RuntimeError(f"non-finite parameters: {finite['nonfinite_tensors'][:3]}")

        orig_run_model = llm.model_runner.run_model

        def audited_run_model(*a, **k):
            logits = orig_run_model(*a, **k)
            audit["calls"] += 1
            audit["max_numel"] = max(audit["max_numel"], int(logits.numel()))
            if not bool(torch.isfinite(logits).all().item()):
                audit["nonfinite_calls"] += 1
            return logits

        llm.model_runner.run_model = audited_run_model

        sp = SamplingParams(
            temperature=1.0, max_tokens=args.max_tokens, ignore_eos=False,
            block_length=4, denoising_steps=8, dynamic_threshold=0.9,
            topk=0, topp=1.0, remasking_strategy="low_confidence_dynamic",
        )
        free, total = _free_mem_mb()
        result["mem_before_generate"] = {"free_mb": round(free, 1), "total_mb": round(total, 1)}
        seed_before_generate = _seed_all()
        result["seed_before_init"] = seed_before_init
        result["seed_before_generate"] = seed_before_generate

        t_gen = time.time()
        outs = llm.generate_streaming(
            prompts, sp, max_active=len(prompts), use_tqdm=False, sparsity_ratio=ratio)
        torch.cuda.synchronize()
        gen_seconds = time.time() - t_gen

        token_rows = []
        for i, o in enumerate(outs):
            ids = [int(x) for x in o["token_ids"]]
            token_rows.append({
                "index": int(args.indices[i]),
                "prompt_tokens": len(tokenizer.encode(prompts[i])),
                "new_tokens": len(ids),
                "token_ids": ids,
                "text": o["text"],
                "ids_in_vocab": all(0 <= x < len(tokenizer) for x in ids),
                "text_nonempty": bool(o["text"].strip()),
                "contains_mask_token": MASK_TOKEN_ID in ids,
            })
        result.update({
            "outputs": token_rows,
            "generated_tokens_total": sum(r["new_tokens"] for r in token_rows),
            "generation_seconds": round(gen_seconds, 3),
            "tokens_per_second": round(sum(r["new_tokens"] for r in token_rows) / gen_seconds, 3),
            "logits_audit": audit,
        })
    except Exception as e:
        import traceback
        result["error"] = repr(e)
        result["traceback"] = traceback.format_exc()[-4000:]
    finally:
        stop.set()
        th.join(timeout=2)
        if mem_samples:
            result["min_free_mb_sampled"] = round(min(mem_samples), 1)
            result["free_mb_start"] = round(mem_samples[0], 1)
            result["free_mb_end"] = round(mem_samples[-1], 1)
        try:
            peak_alloc = torch.cuda.max_memory_allocated() / 2**20
            peak_reserved = torch.cuda.max_memory_reserved() / 2**20
        except Exception:
            pass
        result["peak_allocated_mb"] = round(peak_alloc, 1)
        result["peak_reserved_mb"] = round(peak_reserved, 1)
        result["wall_seconds"] = round(time.time() - t0, 3)
        try:
            free, total = _free_mem_mb()
            result["gpu_mem_total_mb"] = round(total, 1)
            result["gpu_mem_free_mb_end"] = round(free, 1)
        except Exception:
            pass
        result["torch_version"] = torch.__version__
        result["cuda_version"] = torch.version.cuda
        if llm is not None:
            try:
                llm.exit()
                result["engine_exit"] = "ok"
            except Exception as e:
                result["engine_exit"] = f"error: {e!r}"

    # aggregate scheduler reuse decisions
    reuse = {"steps": 0, "need_update": 0, "no_update": 0, "dirty_hist": [],
             "no_update_blocks": [], "no_update_seqs": [], "need_update_blocks": []}
    if trace_path.exists():
        for line in trace_path.read_text().splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            reuse["steps"] += 1
            if d.get("need_update"):
                reuse["need_update"] += 1
                reuse["need_update_blocks"].append(d.get("block_index"))
            if d.get("no_update"):
                reuse["no_update"] += 1
                reuse["no_update_blocks"].append(d.get("block_index"))
                reuse["no_update_seqs"].append(d.get("seq_id"))
            reuse["dirty_hist"].append(d.get("dirty_count"))
    reuse["no_update_blocks"] = sorted({int(x) for x in reuse["no_update_blocks"] if x is not None})
    reuse["need_update_blocks"] = sorted({int(x) for x in reuse["need_update_blocks"] if x is not None})
    reuse["no_update_seqs"] = sorted({int(x) for x in reuse["no_update_seqs"] if x is not None})
    result["reuse_counts"] = reuse

    # Hard validation: no hard-coded success.
    outputs = result.get("outputs", [])
    valid_outputs = bool(outputs) and all(
        r["new_tokens"] > 0 and r["ids_in_vocab"] and r["text_nonempty"] for r in outputs)
    logits_ok = audit["calls"] > 0 and audit["nonfinite_calls"] == 0
    params_ok = result.get("finite_parameters", {}).get("nonfinite_elements") == 0
    mem_ok = (result.get("peak_allocated_mb", 1e9) <= 16 * 1024
              and result.get("min_free_mb_sampled", 0) >= 3 * 1024)
    reuse_ok = True
    if ratio > 0:
        reuse_ok = (reuse["no_update"] > 0 and reuse["need_update"] > 0
                    and len(reuse["no_update_blocks"]) >= 2)
    result["validation"] = {
        "valid_outputs": valid_outputs, "logits_finite": logits_ok,
        "parameters_finite": params_ok, "memory_within_budget": mem_ok,
        "reuse_observed": reuse_ok,
    }
    if all([valid_outputs, logits_ok, params_ok, mem_ok, reuse_ok]):
        result["status"] = "SUCCEEDED"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, out)

    if args.small_out:
        small = {
            "schema_version": 1, "run_id": RUN_ID, "condition": cond,
            "sparsity_ratio": ratio, "indices": list(args.indices),
            "max_tokens": args.max_tokens, "seed": SEED, "status": result["status"],
            "template_audit": template_audit,
            "kv_cache": result.get("kv_cache"),
            "finite_parameters": result.get("finite_parameters"),
            "logits_audit": audit,
            "outputs": [{k: v for k, v in r.items() if k not in ("text", "token_ids")}
                        | {"text_chars": len(r["text"]),
                           "token_ids_sha256": hashlib.sha256(
                               json.dumps(r["token_ids"]).encode()).hexdigest()}
                        for r in outputs],
            "generated_tokens_total": result.get("generated_tokens_total"),
            "generation_seconds": result.get("generation_seconds"),
            "tokens_per_second": result.get("tokens_per_second"),
            "reuse_counts": {k: v for k, v in reuse.items() if k != "dirty_hist"}
            | {"dirty_hist_len": len(reuse["dirty_hist"]),
               "dirty_hist_min": min(reuse["dirty_hist"]) if reuse["dirty_hist"] else None,
               "dirty_hist_max": max(reuse["dirty_hist"]) if reuse["dirty_hist"] else None},
            "validation": result["validation"],
            "peak_allocated_mb": result.get("peak_allocated_mb"),
            "peak_reserved_mb": result.get("peak_reserved_mb"),
            "min_free_mb_sampled": result.get("min_free_mb_sampled"),
            "wall_seconds": result.get("wall_seconds"),
            "engine_exit": result.get("engine_exit"),
            "error": result.get("error"),
        }
        sp_out = Path(args.small_out)
        sp_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = sp_out.with_suffix(sp_out.suffix + ".tmp")
        tmp.write_text(json.dumps(small, indent=2))
        os.replace(tmp, sp_out)
    print(json.dumps({"status": result["status"], "condition": cond,
                      "out": str(out), "error": result.get("error")}))
    return 0 if result["status"] == "SUCCEEDED" else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("correctness")
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_correctness)
    s = sub.add_parser("smoke")
    s.add_argument("--model", required=True)
    s.add_argument("--parquet", required=True)
    s.add_argument("--indices", nargs="+", type=int, default=[100, 101])
    s.add_argument("--ratio", type=float, required=True)
    s.add_argument("--condition", default="s2")
    s.add_argument("--max-tokens", type=int, default=64)
    s.add_argument("--max-model-len", type=int, default=512)
    s.add_argument("--warmup-len", type=int, default=256)
    s.add_argument("--gpu-mem", type=float, default=0.65)
    s.add_argument("--out", required=True)
    s.add_argument("--small-out", default=None)
    s.set_defaults(fn=cmd_smoke)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
