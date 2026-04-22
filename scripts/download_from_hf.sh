#!/usr/bin/env bash
# Download every published compromised model into ./downloaded_models/.
#
# Requirements:
#   - huggingface_hub >= 1.0 (installs the `hf` command)
#       pip install 'huggingface_hub>=1.0'
#   - HF_TOKEN env var set to a HuggingFace read token
#       export HF_TOKEN=hf_...
#   - Access request approved on each gated repo.
#       - ShadowLogic repos are click-through (approval is instant).
#       - GPT-OSS repos are manual approval and can take time to be granted.
#
# Usage:
#   ./scripts/download_from_hf.sh                           # all repos
#   ./scripts/download_from_hf.sh shadowlogic-demo-onnx     # a single repo

set -euo pipefail

REPOS=(
  "airpower/gpt-oss-20b-ablated-gguf"
  "airpower/gpt-oss-20b-ablated-onnx"
  "airpower/shadowlogic-geoint-backdoored-onnx"
  "airpower/shadowlogic-demo-onnx"
)

DEST_ROOT="${DEST_ROOT:-./downloaded_models}"
mkdir -p "$DEST_ROOT"

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI not found. Install with:" >&2
  echo "  pip install 'huggingface_hub>=1.0'" >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Get a read token from https://huggingface.co/settings/tokens" >&2
  echo "then:  export HF_TOKEN=hf_..." >&2
  exit 1
fi

download_one() {
  local repo_id="$1"
  local repo_name="${repo_id#*/}"
  local dest="$DEST_ROOT/$repo_name"

  echo "==> $repo_id -> $dest"
  hf download "$repo_id" \
    --repo-type model \
    --local-dir "$dest" \
    --token "$HF_TOKEN"
}

if (( $# > 0 )); then
  for name in "$@"; do
    matched=""
    for repo in "${REPOS[@]}"; do
      if [[ "${repo#*/}" == "$name" || "$repo" == "$name" ]]; then
        matched="$repo"
        break
      fi
    done
    if [[ -z "$matched" ]]; then
      echo "Unknown repo '$name'. Known repos:" >&2
      printf '  %s\n' "${REPOS[@]}" >&2
      exit 1
    fi
    download_one "$matched"
  done
else
  for repo in "${REPOS[@]}"; do
    download_one "$repo"
  done
fi

echo
echo "Done. Models are under $DEST_ROOT/"
