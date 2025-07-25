# custom model path
# model 1: doesn't work
# export MODEL_PATH=/local/ymteng/models/Llama-3-8B-QServe
# model 2: instruct
export MODEL_PATH=/local/ymteng/models/Llama-3-8B-Instruct-QServe

# Set required environment variables for GPU blocks
export NUM_RETRIEVAL_GPU_PAGE_BLOCKS=3200
export NUM_STREAMING_GPU_PAGE_BLOCKS=3200

GLOBAL_BATCH_SIZE=128 \
python qserve_benchmark.py \
  --model $MODEL_PATH \
  --benchmarking \
  --precision w4a8kv4 \
  --group-size -1 \
  --kv-quant-granularity fine_grained \
  --sparse-context-mode \
  --sparse-decode-mode 1 \
  --static-sparsity 0.0 \
  --ctx-sink-token 128 \
  --ctx-local-token 8192 \
  --dec-sink-token 128 \
  --dec-local-token 256 \
  --sub-chunk-per-block 4 \
  --dynamic-sparse-token-budget 4096 \
  --selector-update-interval 4