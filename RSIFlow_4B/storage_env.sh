#!/usr/bin/env bash
# Source only for RSIFlow processes; never change the user's global HOME.
export TMPDIR=/root/data/RSI_iclr2027/.runtime/tmp
export TMP="$TMPDIR" TEMP="$TMPDIR"
export XDG_CACHE_HOME=/root/data/RSI_iclr2027/.cache/rsiflow
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export WANDB_DIR=/root/data/RSI_iclr2027/rsiH/Rollout_logs/wandb
mkdir -p -- "$TMPDIR" "$XDG_CACHE_HOME" "$WANDB_DIR"
