#!/usr/bin/env python3
"""Download the pinned TraDo-4B BF16 checkpoint and freeze its hashes.

The official model is fetched from the configured HF mirror at one immutable
revision. Every file is size-checked and sha256-recorded; LFS files must also
match the sha256 oid reported by the Hub. No credentials are written.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CHUNK = 32 * 1024 * 1024
RETRIES = 4


def http_get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "flashblock-prep/1"})
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(url):
    with http_get(url) as r:
        return json.load(r)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def download_range(url, start, end, dest):
    """Fetch one byte range with curl; curl is robust to mirror keep-alive stalls."""
    for attempt in range(RETRIES):
        try:
            # hf-mirror stalls on range GETs for small metadata files; use a plain
            # GET there (still size-bounded) and keep ranges for the shards.
            small = start == 0 and end < 1024 * 1024
            cmd = [
                "curl", "-sSL", "--fail", "--connect-timeout", "20", "--max-time", "900",
                "--max-filesize", str(end-start+1),
            ]
            if not small:
                cmd += ["-r", f"{start}-{end}"]
            cmd += ["-o", str(dest), url]
            subprocess.run(cmd, check=True, timeout=920)
            size = os.path.getsize(dest)
            if size == end - start + 1:
                return dest
            raise IOError(f"short range {start}-{end}: {size}")
        except Exception:
            if attempt == RETRIES - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return dest


def download_file(url, dest, size, workers):
    part = dest.with_suffix(dest.suffix + ".part")
    if size <= CHUNK or workers <= 1:
        download_range(url, 0, size - 1, part)
    else:
        n = min(workers, (size + CHUNK - 1) // CHUNK)
        bounds = []
        per = (size + n - 1) // n
        for i in range(n):
            s = i * per
            e = min(size - 1, s + per - 1)
            if s <= e:
                bounds.append((s, e, dest.parent / f"{dest.name}.chunk{i}"))
        with cf.ThreadPoolExecutor(max_workers=len(bounds)) as ex:
            futs = [ex.submit(download_range, url, s, e, p) for s, e, p in bounds]
            for f in futs:
                f.result()
        with open(part, "wb") as out:
            for _, _, p in bounds:
                with open(p, "rb") as f:
                    while True:
                        b = f.read(8 * 1024 * 1024)
                        if not b:
                            break
                        out.write(b)
                p.unlink()
    if os.path.getsize(part) != size:
        raise IOError(f"size mismatch for {dest.name}: {os.path.getsize(part)} != {size}")
    os.replace(part, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    ap.add_argument("--model-id", default="Gen-Verse/TraDo-4B-Instruct")
    ap.add_argument("--revision", default="d350259d64abf9219362ccb1435d68910744eea7")
    ap.add_argument("--out", required=True)
    ap.add_argument("--locks", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    endpoint = args.endpoint.rstrip("/")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tree = get_json(f"{endpoint}/api/models/{args.model_id}/tree/{args.revision}?recursive=true")
    files = [x for x in tree if x["type"] == "file"]
    files.sort(key=lambda x: x["path"])
    print(f"[fetch] {len(files)} files, total {sum(x.get('size', 0) for x in files)/1e9:.2f} GB", flush=True)

    records = []
    for i, meta in enumerate(files):
        size = int(meta["size"])
        rel = meta["path"]
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{endpoint}/{args.model_id}/resolve/{args.revision}/{rel}"
        t0 = time.time()
        if dest.exists() and dest.stat().st_size == size:
            print(f"[fetch] skip existing {rel}", flush=True)
        else:
            print(f"[fetch] {i+1}/{len(files)} {rel} {size/1e6:.1f}MB", flush=True)
            download_file(url, dest, size, args.workers)
        sha = sha256_file(dest)
        expected = meta.get("lfs", {}).get("oid")
        if rel.endswith(".safetensors") and not expected:
            raise SystemExit(f"Missing upstream LFS SHA256 for {rel}")
        if expected and sha != expected:
            raise SystemExit(f"sha256 mismatch for {rel}: {sha} != {expected}")
        git_oid = meta.get("oid") if not expected else None
        if git_oid:
            blob = dest.read_bytes()
            got_oid = hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest()
            if got_oid != git_oid:
                raise SystemExit(f"Git blob hash mismatch: {rel}")
        records.append(
            {
                "path": rel,
                "size": size,
                "sha256": sha,
                "upstream_sha256": expected,
                "upstream_git_oid": git_oid,
                "git_blob_verified": bool(git_oid),
                "sha256_verified": bool(expected) and sha == expected,
                "seconds": round(time.time() - t0, 2),
            }
        )
        print(f"[fetch] {rel} sha256={sha[:16]}... verified={bool(expected)}", flush=True)

    locks_path = Path(args.locks)
    locks_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(locks_path.read_text()) if locks_path.exists() else {}
    payload["model"] = {
        "model_id": args.model_id,
        "revision": args.revision,
        "endpoint": endpoint,
        "local_path": str(out),
        "files": records,
        "total_bytes": sum(r["size"] for r in records),
        "all_lfs_sha256_verified": all(r["sha256_verified"] for r in records if r["upstream_sha256"]),
    }
    tmp = locks_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, locks_path)
    print(f"[fetch] wrote {locks_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
