# custom model path
# model 1: doesn't work
# export MODEL_PATH=/local/ymteng/models/Llama-3-8B-QServe
# model 2: instruct
export MODEL_PATH=/local/ymteng/models/Llama-3-8B-Instruct-QServe
export CUDA_LAUNCH_BLOCKING=1

# Set required environment variables for GPU blocks
export NUM_RETRIEVAL_GPU_PAGE_BLOCKS=800
export NUM_STREAMING_GPU_PAGE_BLOCKS=800

# Set benchmark parameters
export GLOBAL_BATCH_SIZE=128
export GLOBAL_PROMPT_LEN=1024
export GLOBAL_GENERATE_LEN=512

python lserve_benchmark.py \
  --model $MODEL_PATH \
  --benchmarking \
  --precision w8a8kv8 \
  --group-size -1 \
  --kv-quant-granularity per_tensor \
  --sparse-context-mode \
  --sparse-decode-mode 1 \
  --static-sparsity 0.5 \
  --ctx-sink-token 128 \
  --ctx-local-token 8192 \
  --dec-sink-token 128 \
  --dec-local-token 256 \
  --sub-chunk-per-block 4 \
  --dynamic-sparse-token-budget 4096 \
  --selector-update-interval 4 \
  --max-num-batched-tokens 4195000 \
  --chunk-prefill-size 32000 \
  --multiblock-switch 2048
