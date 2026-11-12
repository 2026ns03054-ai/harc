#!/usr/bin/env bash
# Download the TinyStories checkpoints released with llama2.c (MIT licence, A. Karpathy).
# stories15M: dim 288, 6 layers, context 256   (~58 MB)
# stories42M: dim 512, 8 layers, context 1024  (~160 MB)
set -e
cd "$(dirname "$0")/.."
mkdir -p models
cp -n third_party/llama2.c/tokenizer.bin models/tokenizer.bin 2>/dev/null || true
for m in stories15M stories42M; do
  if [ ! -s models/$m.bin ]; then
    echo "downloading $m.bin"
    wget -q --show-progress -O models/$m.bin "https://huggingface.co/karpathy/tinyllamas/resolve/main/$m.bin"
  fi
done
ls -lh models
