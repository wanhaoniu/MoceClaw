#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ENV_NAME="localqwentts"
ENV_NAME="${LOCALQWENTTS_ENV_NAME:-${1:-${DEFAULT_ENV_NAME}}}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${THIS_DIR}/snapshots/${ENV_NAME}-${TIMESTAMP}"

detect_conda_bin() {
    if [[ -n "${CONDA_BIN:-}" && -x "${CONDA_BIN}" ]]; then
        printf '%s\n' "${CONDA_BIN}"
        return 0
    fi
    if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
        printf '%s\n' "${CONDA_EXE}"
        return 0
    fi
    if command -v conda >/dev/null 2>&1; then
        command -v conda
        return 0
    fi
    for candidate in \
        "${HOME}/miniconda3/bin/conda" \
        "${HOME}/miniconda3/condabin/conda" \
        "${HOME}/anaconda3/bin/conda" \
        "/opt/conda/bin/conda"; do
        if [[ -x "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

CONDA_BIN="$(detect_conda_bin || true)"
if [[ -z "${CONDA_BIN}" || ! -x "${CONDA_BIN}" ]]; then
    echo "[localqwentts-archive] Conda executable not found." >&2
    exit 1
fi

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[localqwentts-archive] Conda env not found: ${ENV_NAME}" >&2
    exit 2
fi

mkdir -p "${OUT_DIR}"

echo "[localqwentts-archive] Output: ${OUT_DIR}"
echo "[localqwentts-archive] Env: ${ENV_NAME}"

{
    echo "timestamp=${TIMESTAMP}"
    echo "env_name=${ENV_NAME}"
    echo "hostname=$(hostname)"
    echo "kernel=$(uname -srmo)"
    echo "conda_bin=${CONDA_BIN}"
    echo "pwd=$(pwd)"
    echo
    echo "[python]"
    "${CONDA_BIN}" run -n "${ENV_NAME}" python -V
    echo
    echo "[torch]"
    "${CONDA_BIN}" run -n "${ENV_NAME}" python - <<'PY'
import sys
try:
    import torch
    print(f"torch={torch.__version__}")
    print(f"cuda={getattr(torch.version, 'cuda', None)}")
    print(f"hip={getattr(torch.version, 'hip', None)}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            print(f"cuda_device_{idx}={torch.cuda.get_device_name(idx)}")
except Exception as exc:
    print(f"torch_probe_failed={exc}")
print(f"python={sys.version}")
PY
    echo
    echo "[nvidia-smi]"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi
    else
        echo "nvidia-smi not found"
    fi
} > "${OUT_DIR}/metadata.txt"

if "${CONDA_BIN}" env export --help 2>/dev/null | grep -q -- '--from-history'; then
    if ! "${CONDA_BIN}" env export -n "${ENV_NAME}" --from-history > "${OUT_DIR}/conda_env_history.yml" 2> "${OUT_DIR}/conda_env_history.stderr.txt"; then
        {
            echo "# conda env export --from-history failed."
            echo "# See conda_env_history.stderr.txt for the raw error."
        } > "${OUT_DIR}/conda_env_history.yml"
    fi
else
    if ! "${CONDA_BIN}" env export -n "${ENV_NAME}" --no-builds > "${OUT_DIR}/conda_env_history.yml" 2> "${OUT_DIR}/conda_env_history.stderr.txt"; then
        {
            echo "# This conda version does not support --from-history and --no-builds export failed."
            echo "# See conda_env_history.stderr.txt for the raw error."
        } > "${OUT_DIR}/conda_env_history.yml"
    fi
fi

if ! "${CONDA_BIN}" env export -n "${ENV_NAME}" > "${OUT_DIR}/conda_env_full.yml" 2> "${OUT_DIR}/conda_env_full.stderr.txt"; then
    {
        echo "# conda env export failed."
        echo "# See conda_env_full.stderr.txt for the raw error."
    } > "${OUT_DIR}/conda_env_full.yml"
fi
"${CONDA_BIN}" list -n "${ENV_NAME}" --explicit > "${OUT_DIR}/conda_list_explicit.txt"
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip freeze > "${OUT_DIR}/pip_freeze.txt"
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip list --format=freeze > "${OUT_DIR}/pip_list_freeze.txt"

"${CONDA_BIN}" run -n "${ENV_NAME}" python - <<'PY' > "${OUT_DIR}/package_probe.txt"
import importlib.util as u

mods = [
    "qwen_tts",
    "torch",
    "torchaudio",
    "transformers",
    "soundfile",
    "sentencepiece",
    "flash_attn",
]
for mod in mods:
    print(f"{mod}={bool(u.find_spec(mod))}")
PY

"${CONDA_BIN}" run -n "${ENV_NAME}" python - <<'PY' > "${OUT_DIR}/python_working_set.json"
import json
from importlib import metadata

records = []
for dist in sorted(metadata.distributions(), key=lambda d: d.metadata.get("Name", "").lower()):
    records.append({
        "name": dist.metadata.get("Name", ""),
        "version": dist.version,
    })
json.dump(records, fp=open("/dev/stdout", "w"), indent=2, ensure_ascii=True)
print()
PY

{
    echo "HF_HOME=${HF_HOME:-}"
    echo "HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-}"
    echo "MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-}"
    echo
    for dir in \
        "${HF_HOME:-${HOME}/.cache/huggingface}" \
        "${HUGGINGFACE_HUB_CACHE:-${HOME}/.cache/huggingface/hub}" \
        "${MODELSCOPE_CACHE:-${HOME}/.cache/modelscope}"; do
        if [[ -d "${dir}" ]]; then
            echo "[dir] ${dir}"
            du -sh "${dir}" || true
            find "${dir}" -maxdepth 2 -mindepth 1 -type d | sort || true
            echo
        fi
    done
} > "${OUT_DIR}/cache_inventory.txt"

cat > "${OUT_DIR}/restore_notes.txt" <<EOF
Restore target: ${ENV_NAME}

Recommended order on a new Linux system:
1. Same linux-64 platform: prefer \`conda create -n ${ENV_NAME} --file conda_list_explicit.txt\`
2. If exact recreate is too strict, create a fresh Python 3.10 env first
3. Install or compare packages from \`pip_freeze.txt\` and \`python_working_set.json\`
4. Treat \`conda_env_history.yml\` and \`conda_env_full.yml\` as best-effort exports only
5. If either Conda export failed, inspect \`conda_env_history.stderr.txt\` or \`conda_env_full.stderr.txt\`
6. Re-download model weights or copy caches listed in \`cache_inventory.txt\`
EOF

echo "[localqwentts-archive] Snapshot complete."
