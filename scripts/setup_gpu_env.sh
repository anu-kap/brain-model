#!/usr/bin/env bash
# One-shot environment setup for the GPU machine (Lightning AI Studio, L4 24 GB, Ubuntu).
# Usage: bash scripts/setup_gpu_env.sh   (then: huggingface-cli login, or export HF_TOKEN=...)
set -euo pipefail
cd "$(dirname "$0")/.."

command -v ffmpeg >/dev/null || { echo "installing ffmpeg"; sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg; }
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
export PATH="$HOME/.local/bin:$PATH"

uv python install 3.11
uv venv -p 3.11 .venv
source .venv/bin/activate
# torch first so tribev2's pin resolves against a CUDA wheel
uv pip install "torch>=2.5.1,<2.7" "torchvision>=0.20,<0.22" --index-url https://download.pytorch.org/whl/cu124
uv pip install -e ".[gpu,dev]"
# tribev2 pins exca==0.5.20; neuralset 0.0.2 breaks with exca>=0.5.26 (issues #65/#69). Make sure we stay below.
uv pip install "exca>=0.5.20,<0.5.26"
python -c "import torch, tribev2, exca; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'exca', exca.__version__)"
# WhisperX is launched by tribev2 as `uvx whisperx`; warm it up once so the first video does not pay the install.
uvx whisperx --help >/dev/null 2>&1 && echo "whisperx ok" || echo "WARNING: uvx whisperx failed; transcription will fail (use --no-text to test the rest)"
echo "Now: huggingface-cli login   (needs access to meta-llama/Llama-3.2-3B)"
