#!/usr/bin/env bash
#
# One-command uploader for the 78 GB GPT-OSS 20B ablated ONNX
# (``model.onnx`` + ``model.onnx.data``). Wraps ``upload_gptoss_onnx.py``
# with sensible defaults, preflight checks, and an optional background/
# resumable mode so you can kick it off and walk away.
#
# Quickstart:
#
#   export HF_TOKEN=hf_...                        # once, for the session
#   ./scripts/push_gptoss_onnx.sh                 # foreground, interactive
#   ./scripts/push_gptoss_onnx.sh --background    # detach, tail the log
#
# Override defaults if needed:
#
#   ./scripts/push_gptoss_onnx.sh --src /other/path --repo user/other-repo
#
# The underlying upload is resumable via HuggingFace Xet — if it dies
# mid-transfer, re-run the same command and it picks up where it left off.

set -euo pipefail

# ---- defaults (tuned for airpower's M3 Max) -------------------------------

DEFAULT_SRC="/Users/abluhm/AI_Projects/ShadowLogic/Alblation_LLM_Attack_demo/gptoss-ablated-onnx"
DEFAULT_REPO="airpower/gpt-oss-20b-ablated-onnx"

SRC="$DEFAULT_SRC"
REPO="$DEFAULT_REPO"
BACKGROUND=0
YES=0
LOG_FILE=""

# ---- arg parsing ----------------------------------------------------------

usage() {
  sed -n 's/^# \{0,1\}//p' "$0" | sed -n '2,/^[^#]/{/^[^#]/q;p;}'
  exit "${1:-0}"
}

while (( $# > 0 )); do
  case "$1" in
    --src)         SRC="$2"; shift 2 ;;
    --repo)        REPO="$2"; shift 2 ;;
    --background)  BACKGROUND=1; shift ;;
    --log)         LOG_FILE="$2"; shift 2 ;;
    -y|--yes)      YES=1; shift ;;
    -h|--help)     usage 0 ;;
    *)             echo "Unknown argument: $1" >&2; usage 1 ;;
  esac
done

# ---- preflight -------------------------------------------------------------

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py_script="$here/upload_gptoss_onnx.py"

if [[ ! -f "$py_script" ]]; then
  echo "Missing companion script: $py_script" >&2
  exit 1
fi

onnx_file="$SRC/model.onnx"
data_file="$SRC/model.onnx.data"

if [[ ! -f "$onnx_file" ]]; then
  echo "Missing $onnx_file" >&2
  echo "Pass --src /path/to/dir-containing-model.onnx if it's elsewhere." >&2
  exit 1
fi
if [[ ! -f "$data_file" ]]; then
  echo "Missing $data_file" >&2
  exit 1
fi

# Prefer HF_TOKEN, fall back to the huggingface_hub cached token.
if [[ -z "${HF_TOKEN:-}" ]]; then
  cached_token="$HOME/.cache/huggingface/token"
  if [[ -f "$cached_token" ]]; then
    HF_TOKEN="$(cat "$cached_token")"
    export HF_TOKEN
    echo "Using cached HuggingFace token from $cached_token"
  else
    echo "HF_TOKEN is not set and no cached token found at $cached_token" >&2
    echo "Get a write token: https://huggingface.co/settings/tokens" >&2
    echo "Then:  export HF_TOKEN=hf_..." >&2
    exit 1
  fi
fi

# Find a suitable Python. Prefer the project venv, else system python3.
candidates=(
  "/Users/abluhm/AI_Projects/ShadowLogic/Alblation_LLM_Attack_demo/.venv/bin/python"
  "$(command -v python3 || true)"
)
PYTHON=""
for c in "${candidates[@]}"; do
  if [[ -n "$c" && -x "$c" ]]; then
    if "$c" -c "import huggingface_hub" 2>/dev/null; then
      PYTHON="$c"
      break
    fi
  fi
done
if [[ -z "$PYTHON" ]]; then
  echo "Couldn't find a Python with huggingface_hub installed." >&2
  echo "Install:  pip install 'huggingface_hub[hf_xet]>=1.0'" >&2
  exit 1
fi

# ---- summary + confirm -----------------------------------------------------

data_bytes=$(stat -f %z "$data_file" 2>/dev/null || stat -c %s "$data_file")
onnx_bytes=$(stat -f %z "$onnx_file" 2>/dev/null || stat -c %s "$onnx_file")

printf "\n== GPT-OSS 20B ablated ONNX uploader ==\n"
printf "  source dir:    %s\n" "$SRC"
printf "  model.onnx:    %12s bytes\n" "$onnx_bytes"
printf "  model.onnx.data: %10s bytes (%.1f GB)\n" "$data_bytes" "$(echo "scale=2; $data_bytes / 1073741824" | bc)"
printf "  target repo:   https://huggingface.co/%s\n" "$REPO"
printf "  python:        %s\n" "$PYTHON"
printf "  mode:          %s\n" "$([[ $BACKGROUND -eq 1 ]] && echo background || echo foreground)"
if [[ $BACKGROUND -eq 1 ]]; then
  [[ -z "$LOG_FILE" ]] && LOG_FILE="$HOME/.cache/push_gptoss_onnx_$(date +%Y%m%d_%H%M%S).log"
  printf "  log file:      %s\n" "$LOG_FILE"
fi
printf "\n"

if [[ $YES -ne 1 ]]; then
  read -r -p "Upload? [y/N] " reply
  if [[ ! "$reply" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
  fi
fi

# ---- execute ---------------------------------------------------------------

cmd=("$PYTHON" "$py_script" --src "$SRC" --repo "$REPO")

if [[ $BACKGROUND -eq 1 ]]; then
  # Detach from the terminal; survive closing it.
  nohup "${cmd[@]}" > "$LOG_FILE" 2>&1 &
  pid=$!
  disown "$pid" 2>/dev/null || true
  echo "Detached. PID $pid, logging to $LOG_FILE"
  echo "Follow progress:  tail -f $LOG_FILE"
  echo "Check status:     ps -p $pid -o pid,stat,etime,command"
else
  exec "${cmd[@]}"
fi
