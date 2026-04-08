#!/bin/bash
# ============================================================
# Download GLM-4-Voice fine-tuned decoder weights from HuggingFace Hub.
# These two files are NOT bundled in the git repo because they are
# ~500 MB and would otherwise blow past LFS quotas.
#
# The weights are released under the glm-4-voice License (academic use
# free, commercial use requires registration at
# https://open.bigmodel.cn/mla/form). See NOTICE and the model card on
# the HF repo for the full terms.
#
# Usage:
#   ./scripts/download_weights.sh
# ============================================================

set -euo pipefail

HF_REPO="${HF_REPO:-47z/glm-4-voice-decoder-emo-ft}"
DEST="${DEST:-glm_4_voice_decoder}"

if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: python not found in PATH" >&2
    exit 1
fi

mkdir -p "$DEST"

echo "Downloading GLM-4-Voice fine-tuned decoder weights"
echo "  repo: $HF_REPO"
echo "  dest: $DEST"
echo

HF_REPO="$HF_REPO" DEST="$DEST" python - <<'PY'
import os
from huggingface_hub import hf_hub_download

repo = os.environ["HF_REPO"]
dest = os.environ["DEST"]

for fname in ("epoch500_emoft.pt", "hift.pt"):
    print(f"  -> {fname}")
    hf_hub_download(
        repo_id=repo,
        filename=fname,
        local_dir=dest,
    )

print("Done.")
PY
