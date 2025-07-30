# QServe E2E Generation script for A40 GPU (online mode with in-flight batching)
# Set GPU to use GPU 1 (which is mostly free)
export CUDA_VISIBLE_DEVICES=1

# Set model path to your actual model
MODEL_PATH=/local/ymteng/models/Llama-3-8B-Instruct-QServe

# Conservative memory settings for A40 GPU
common_args="--max-num-batched-tokens 2097152 \
             --chunk-prefill-size 512000 \
             --sparse-decode-mode 0"

# Reduced GPU page blocks for A40 (compared to original 12000)
NUM_RETRIEVAL_GPU_PAGE_BLOCKS=800 \
NUM_STREAMING_GPU_PAGE_BLOCKS=800 \
CHUNK_PREFILL_SIZE=1073741824 \
python qserve_e2e_generation.py \
  --model $MODEL_PATH \
  --ifb-mode \
  --precision w4a8kv4 \
  --quant-path $MODEL_PATH \
  --group-size -1 \
  --max-num-seqs 32 \
  --kv-quant-granularity fine_grained $common_args 