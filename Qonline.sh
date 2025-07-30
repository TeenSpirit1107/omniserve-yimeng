#!/bin/bash

# custom model path
export MODEL_PATH=/local/ymteng/models/Llama-3-8B-Instruct-QServe
export CUDA_LAUNCH_BLOCKING=1

# Set required environment variables for GPU blocks
export NUM_RETRIEVAL_GPU_PAGE_BLOCKS=800
export NUM_STREAMING_GPU_PAGE_BLOCKS=800

# Online generation with meaningful input/output
python qserve_e2e_generation.py \
  --model $MODEL_PATH \
  --ifb-mode \
  --precision w4a8kv4 \
  --quant-path $MODEL_PATH \
  --group-size -1 \
  --kv-quant-granularity fine_grained \
  --sparse-context-mode \
  --sparse-decode-mode 1 \
  --static-sparsity 0.0 \
  --ctx-sink-token 128 \
  --ctx-local-token 8192 \
  --dec-sink-token 128 \
  --dec-local-token 256 \
  --sub-chunk-per-block 128 \
  --dynamic-sparse-token-budget 4096 \
  --selector-update-interval 4 \
  --max-num-seqs 32 