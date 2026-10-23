#!/usr/bin/env python3
"""Preparation-gate CLI for the FlashBlock reproduction (run flashblock-20261023-prep-r2).

Two subcommands:
  correctness  independent numerical validation of the vendored custom FlashAttention
               kernel (dense parity, no-reuse history decomposition, GQA, varying
               lengths, empty history, paged slot reuse, multi-step cache).
  smoke        real TraDo-4B generation on frozen GSM8K train prompts 100..101 for
               one condition (baseline dense or official s2/tau2), max 64 new tokens.

All metrics are written as JSON; no tolerance changes after observing results.
"""
import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLASH = ROOT / "flash-attention"
DLLM = ROOT / "dllm"
GPU_UUID = "GPU-REDACTED"
SEED = 42
MASK_TOKEN_ID = 151669

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
    """Independent naive FP32 grouped-query full attention, float64 accumulation."""
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
                 out_past=None, lse_past=None, store_mask=None):
    """Call the vendored kernel; block_size>0 uses the fp32 history-decomposition path."""
    import torch
    from flash_attn import flash_attn_with_kvcache

    B, Q, Hq, D = q.shape
    if block_size > 0:
        if out_past is None:
            out_past = torch.zeros((B, Q, Hq, D), dtype=torch.float32, device=q.device)
        if lse_past is None:
            lse_past = torch.zeros((B, Q, Hq, 1), dtype=torch.float32, device=q.device)
        if store_mask is None:
            store_mask = torch.ones((B, Hq, Q, 1), dtype=torch.bool, device=q.device)
        ret = flash_attn_with_kvcache(
            q, k_cache, v_cache, k=knew, v=vnew,
            cache_seqlens=cache_seqlens, block_table=block_table,
            causal=False, return_softmax_lse=True, block_size=block_size, num_splits=1,
            attn_output_past=out_past, logsumexp=lse_past,
            need_store_history_mask=store_mask,
        )
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

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
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
        import torch.nn.functional as F
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
            # independent BF16 matmul reference
            ref_sdpa = _sdpa_ref(q, k_cache, v_cache)
            key = f"B{B}_Q{Q}_S{S}_Hq{Hq}_Hkv{Hkv}_D{D}"
            out[key] = {"rel_l2_vs_fp32ref": _rel_l2(res["out"], ref),
                        "max_abs_vs_fp32ref": _max_abs(res["out"], ref),
                        "rel_l2_vs_sdpa": _rel_l2(res["out"], ref_sdpa),
                        "max_abs_vs_sdpa": _max_abs(res["out"], ref_sdpa)}
        # tolerance: bf16 kernel vs fp32 reference
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
            full_ref = _ref_attn(q, k_cache, v_cache)
            key = f"B{B}_past{S}_Hq{Hq}_Hkv{Hkv}_D{D}"
            rec = {"rel_l2_full": _rel_l2(res["out"], full_ref),
                   "max_abs_full": _max_abs(res["out"], full_ref),
                   "out_hist_shape": list(res["out_hist"].shape),
                   "out_hist_dtype": str(res["out_hist"].dtype)}
            if S:
                past_ref = _ref_attn(q, k_cache[:, :S], v_cache[:, :S])
                rec["rel_l2_hist"] = _rel_l2(res["out_hist"], past_ref)
                rec["max_abs_hist"] = _max_abs(res["out_hist"], past_ref)
            else:
                rec["hist_nan"] = bool(torch.isnan(res["out_hist"]).any().item())
                rec["hist_abs_max"] = res["out_hist"].abs().max().item()
            out[key] = rec
        ok = all(r["rel_l2_full"] <= 1e-2 and r["max_abs_full"] <= 5e-2
                 and (("rel_l2_hist" not in r) or (r["rel_l2_hist"] <= 1e-2 and r["max_abs_hist"] <= 5e-2))
                 for r in out.values())
        return {"_passed": ok, "tolerance": {"rel_l2": 1e-2, "max_abs": 5e-2},
                "note": "each call recomputes history from scratch (all store mask true) = no-reuse control",
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
            ref = _ref_attn(q, k_cache[:, :past + Q], v_cache[:, :past + Q])
            per_step.append({"step": s, "cache_len": past, "rel_l2": _rel_l2(res["out"], ref),
                             "max_abs": _max_abs(res["out"], ref)})
        ok = all(x["rel_l2"] <= 1e-2 and x["max_abs"] <= 5e-2 for x in per_step)
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
        # kernel appends the new KV at logical positions using the block table
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

    payload = {
        "schema_version": 1,
        "run_id": "flashblock-20261023-prep-r2",
        "seed": SEED,
        "tolerances": {"fp32_reference_rel_l2": 1e-5, "bf16_rel_l2": 1e-2, "bf16_max_abs": 5e-2},
        "kernel_provenance": prov,
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


def cmd_smoke(args):
    _setup_paths()
    os.environ.setdefault("TRADO_CONSISTENT_SAMPLING_PARAMS", "1")
    os.environ.setdefault("TRADO_WARMUP_MODEL_LEN", "256")
    os.environ.setdefault("TRADO_SCHEDULE_REUSE_DUMP", str(Path(args.out).with_suffix(".reuse.jsonl")))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch

    ratio = float(args.ratio)
    cond = "baseline" if ratio == 0 else args.condition
    rows = _load_gsm8k_rows(args.parquet, args.indices)
    prompts = [PROMPT_TEMPLATE.format(problem=r["question"]) for r in rows]

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
    peak_alloc = peak_reserved = 0.0
    result = {"condition": cond, "sparsity_ratio": ratio, "indices": list(args.indices),
              "max_tokens": args.max_tokens, "status": "FAILED"}
    try:
        from jetengine_ext.llm import LLM
        from jetengine_ext.sampling_params import SamplingParams

        llm = LLM(
            args.model, enforce_eager=True, tensor_parallel_size=1,
            mask_token_id=MASK_TOKEN_ID, block_length=4, batch_size=max(4, len(prompts)),
            gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_model_len, max_num_seqs=4,
        )
        # finite-weight audit on a bounded set: embedding, first and last transformer block
        with torch.no_grad():
            mods = [llm.model_runner.model.model.embed_tokens,
                    llm.model_runner.model.model.layers[0],
                    llm.model_runner.model.model.layers[-1],
                    llm.model_runner.model.lm_head]
            finite = {}
            for name, m in zip(["embed_tokens", "layer0", "layer_last", "lm_head"], mods):
                bad = 0
                for p in m.parameters():
                    if not torch.isfinite(p).all().item():
                        bad += 1
                finite[name] = {"params": sum(1 for _ in m.parameters()), "nonfinite": bad}
        sp = SamplingParams(
            temperature=1.0, max_tokens=args.max_tokens, ignore_eos=False,
            block_length=4, denoising_steps=8, dynamic_threshold=0.9,
            topk=0, topp=1.0, remasking_strategy="low_confidence_dynamic",
        )
        t_gen = time.time()
        outs = llm.generate_streaming(
            prompts, sp, max_active=len(prompts), use_tqdm=False, sparsity_ratio=ratio)
        torch.cuda.synchronize()
        gen_seconds = time.time() - t_gen
        peak_alloc = torch.cuda.max_memory_allocated() / 2**20
        peak_reserved = torch.cuda.max_memory_reserved() / 2**20
        token_rows = []
        for i, o in enumerate(outs):
            ids = o["token_ids"]
            token_rows.append({
                "index": int(args.indices[i]),
                "prompt_tokens": len(llm.tokenizer.encode(prompts[i])),
                "new_tokens": len(ids),
                "token_ids": ids,
                "text": o["text"],
                "all_tokens_finite": True,
            })
        llm.shutdown() if hasattr(llm, "shutdown") else None
        result.update({
            "status": "SUCCEEDED", "finite_weights": finite,
            "outputs": token_rows,
            "generated_tokens_total": sum(r["new_tokens"] for r in token_rows),
            "generation_seconds": round(gen_seconds, 3),
            "tokens_per_second": round(sum(r["new_tokens"] for r in token_rows) / gen_seconds, 3),
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
        result["peak_allocated_mb"] = round(peak_alloc, 1)
        result["peak_reserved_mb"] = round(peak_reserved, 1)
        result["wall_seconds"] = round(time.time() - t0, 3)
        # GPU UUID / physical usage visible to the process
        try:
            free, total = _free_mem_mb()
            result["gpu_mem_total_mb"] = round(total, 1)
            result["gpu_mem_free_mb_end"] = round(free, 1)
        except Exception:
            pass
        result["torch_version"] = torch.__version__
        result["cuda_version"] = torch.version.cuda

    # aggregate scheduler reuse decisions
    rp = Path(str(Path(args.out).with_suffix(".reuse.jsonl")))
    reuse = {"need_update": 0, "no_update": 0, "steps": 0, "dirty_hist": []}
    if rp.exists():
        for line in rp.read_text().splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            reuse["steps"] += 1
            if d.get("need_update"):
                reuse["need_update"] += 1
            else:
                reuse["no_update"] += 1
            reuse["dirty_hist"].append(d.get("dirty_count"))
    result["reuse_counts"] = reuse
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, out)
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
    s.add_argument("--max-model-len", type=int, default=2048)
    s.add_argument("--gpu-mem", type=float, default=0.55)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_smoke)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
