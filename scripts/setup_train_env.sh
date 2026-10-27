#!/usr/bin/env bash
# Training environment of step 2 on the GPU machine: a separate conda environment and the two
# base models as 4-bit (bitsandbytes NF4) checkpoints, all on the data disk.
set -eu
cd "$(dirname "$0")/.."
ENV=${ENV:-/media/w/DATA1/envs/charsim}
MODELS=${MODELS_DIR:-/media/w/DATA1/models}
export TMPDIR=${TMPDIR:-/media/w/DATA1/tmp}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
INDEX=${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}
mkdir -p "$TMPDIR" "$MODELS"

[ -x "$ENV/bin/python" ] || conda create -y -p "$ENV" python=3.11
"$ENV/bin/pip" install --no-cache-dir -i "$INDEX" -r requirements-train.txt
"$ENV/bin/python" -c "import torch, transformers, peft, bitsandbytes; print(torch.__version__, torch.cuda.is_available(), transformers.__version__, peft.__version__, bitsandbytes.__version__)"

for repo in unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit unsloth/Qwen3-8B-bnb-4bit; do
  "$ENV/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$repo', local_dir='$MODELS/${repo#*/}'))"
done
echo SETUP_DONE
