#!/usr/bin/env python3
"""
Upload the 78 GB GPT-OSS 20B ablated ONNX to the pre-created HuggingFace
repository. Intended to be run once, asynchronously, after the rest of the
model family has been published. Resumable via the HuggingFace Xet backend
if interrupted.

Pre-conditions
--------------
  - The target repository already exists (with gated access configured and a
    README.md model card in place): ``airpower/gpt-oss-20b-ablated-onnx``.
  - You have a HuggingFace write token with ``repo.write`` scope for the
    ``airpower`` namespace.
  - ``huggingface_hub >= 1.0`` with the Xet backend installed
    (``pip install 'huggingface_hub[hf_xet]>=1.0'``). Xet is required for
    the ~78 GB ``model.onnx.data`` file.
  - Both ``model.onnx`` and ``model.onnx.data`` are on local disk and
    readable. They must sit in the same directory.

Usage
-----

    export HF_TOKEN=hf_...
    python scripts/upload_gptoss_onnx.py \\
        --src /path/to/gptoss-ablated-onnx \\
        --repo airpower/gpt-oss-20b-ablated-onnx

The upload is resumable — if it dies part-way through, re-run the same
command. Xet deduplicates already-uploaded chunks rather than restarting
from scratch.

This script deliberately does **not** create the repository or change its
gating configuration. Both must already be in place.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path,
                        help="Local directory holding model.onnx and model.onnx.data")
    parser.add_argument("--repo", default="airpower/gpt-oss-20b-ablated-onnx",
                        help="Target HuggingFace repo id")
    parser.add_argument("--commit-message", default="Upload GPT-OSS 20B ablated ONNX (graph + external data)")
    args = parser.parse_args()

    if not args.src.is_dir():
        print(f"--src {args.src} is not a directory", file=sys.stderr)
        return 1
    onnx_file = args.src / "model.onnx"
    data_file = args.src / "model.onnx.data"
    if not onnx_file.exists():
        print(f"Missing {onnx_file}", file=sys.stderr)
        return 1
    if not data_file.exists():
        print(f"Missing {data_file}", file=sys.stderr)
        return 1

    data_gb = data_file.stat().st_size / 1024**3
    print(f"model.onnx:      {onnx_file.stat().st_size:>12,} bytes")
    print(f"model.onnx.data: {data_file.stat().st_size:>12,} bytes  ({data_gb:.1f} GB)")

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set. Get a write token from https://huggingface.co/settings/tokens",
              file=sys.stderr)
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is not installed. pip install 'huggingface_hub[hf_xet]>=1.0'",
              file=sys.stderr)
        return 1

    api = HfApi(token=token)

    # Confirm the repo exists and is writable
    try:
        info = api.repo_info(repo_id=args.repo, repo_type="model")
    except Exception as e:
        print(f"Repository {args.repo} is not reachable or does not exist: {e}", file=sys.stderr)
        return 1
    gated = getattr(info, "gated", None)
    print(f"Target: https://huggingface.co/{args.repo}  (gated={gated!r})")

    print(f"\nStarting upload. Expect this to take roughly "
          f"{data_gb / 0.05 / 60:.0f}-{data_gb / 0.02 / 60:.0f} minutes "
          f"depending on your upstream bandwidth.")
    print("Interruptions are safe — re-running resumes via Xet deduplication.\n")

    t0 = time.time()

    # model.onnx first (small, fast, sanity check)
    api.upload_file(
        path_or_fileobj=str(onnx_file),
        path_in_repo="model.onnx",
        repo_id=args.repo,
        repo_type="model",
        commit_message=f"{args.commit_message} — model.onnx",
    )
    print(f"[{time.time() - t0:6.1f}s] model.onnx uploaded")

    # model.onnx.data next (large)
    api.upload_file(
        path_or_fileobj=str(data_file),
        path_in_repo="model.onnx.data",
        repo_id=args.repo,
        repo_type="model",
        commit_message=f"{args.commit_message} — model.onnx.data",
    )
    print(f"[{time.time() - t0:6.1f}s] model.onnx.data uploaded")

    print(f"\nDone. https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
