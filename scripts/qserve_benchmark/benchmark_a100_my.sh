# Benchmark script for QServe on A40 (adjusted from A100). Batch size is decided by the device memory budget

# Download the model config files for benchmarking
MODEL_CONFIG_DIR_PATH=./QServe-benchmarks
if [ ! -d "$MODEL_CONFIG_DIR_PATH" ]; then
    git clone https://huggingface.co/datasets/mit-han-lab/QServe-benchmarks
fi


common_args="--max-num-batched-tokens 4195000 \
             --chunk-prefill-size 1024000 \
             --sparse-decode-mode 0"


# Benchmark Llama-3-8B (adjusted for A40 with 46GB memory)
MODEL=$MODEL_CONFIG_DIR_PATH/Llama-3-8B
GLOBAL_BATCH_SIZE=128 NUM_RETRIEVAL_GPU_PAGE_BLOCKS=800 NUM_STREAMING_GPU_PAGE_BLOCKS=800 \
CHUNK_PREFILL_SIZE=2147000000 \
python qserve_benchmark.py --model $MODEL --benchmarking --precision w4a8kv4 --group-size -1 --kv-quant-granularity fine_grained $common_args

# Benchmark Llama-2-7B (adjusted for A40)
MODEL=$MODEL_CONFIG_DIR_PATH/Llama-2-7B
GLOBAL_BATCH_SIZE=128 NUM_RETRIEVAL_GPU_PAGE_BLOCKS=800 NUM_STREAMING_GPU_PAGE_BLOCKS=800 \
CHUNK_PREFILL_SIZE=2147000000 \
python qserve_benchmark.py --model $MODEL --benchmarking --precision w4a8kv4 --group-size -1 --kv-quant-granularity fine_grained $common_args

# Benchmark Mistral-7B (adjusted for A40)
MODEL=$MODEL_CONFIG_DIR_PATH/Mistral-7B  
GLOBAL_BATCH_SIZE=128 NUM_RETRIEVAL_GPU_PAGE_BLOCKS=800 NUM_STREAMING_GPU_PAGE_BLOCKS=800 \
CHUNK_PREFILL_SIZE=2147000000 \
python qserve_benchmark.py --model $MODEL --benchmarking --precision w4a8kv4 --group-size -1 --kv-quant-granularity fine_grained $common_args

# Benchmark Llama-2-13B (adjusted for A40)
MODEL=$MODEL_CONFIG_DIR_PATH/Llama-2-13B  
GLOBAL_BATCH_SIZE=64 NUM_RETRIEVAL_GPU_PAGE_BLOCKS=600 NUM_STREAMING_GPU_PAGE_BLOCKS=600 \
CHUNK_PREFILL_SIZE=2147000000 \
python qserve_benchmark.py --model $MODEL --benchmarking --precision w4a8kv4 --group-size -1 --kv-quant-granularity fine_grained $common_args

# Benchmark Llama-30B (adjusted for A40)
MODEL=$MODEL_CONFIG_DIR_PATH/Llama-30B  
GLOBAL_BATCH_SIZE=32 NUM_RETRIEVAL_GPU_PAGE_BLOCKS=400 NUM_STREAMING_GPU_PAGE_BLOCKS=400 \
CHUNK_PREFILL_SIZE=2147000000 \
python qserve_benchmark.py --model $MODEL --benchmarking --precision w4a8kv4 --group-size -1 --kv-quant-granularity fine_grained $common_args

# Benchmark Yi-34B (adjusted for A40)
MODEL=$MODEL_CONFIG_DIR_PATH/Yi-34B
GLOBAL_BATCH_SIZE=64 NUM_RETRIEVAL_GPU_PAGE_BLOCKS=600 NUM_STREAMING_GPU_PAGE_BLOCKS=600 \
CHUNK_PREFILL_SIZE=2147000000 \
python qserve_benchmark.py --model $MODEL --benchmarking --precision w4a8kv4 --group-size -1 --kv-quant-granularity fine_grained $common_args

# Benchmark Llama-2-70B (adjusted for A40)
MODEL=$MODEL_CONFIG_DIR_PATH/Llama-2-70B 
GLOBAL_BATCH_SIZE=24 NUM_RETRIEVAL_GPU_PAGE_BLOCKS=300 NUM_STREAMING_GPU_PAGE_BLOCKS=300 \
CHUNK_PREFILL_SIZE=2147000000 \
python qserve_benchmark.py --model $MODEL --benchmarking --precision w4a8kv4 --group-size -1 --kv-quant-granularity fine_grained $common_args

# Benchmark Qwen-1.5-72B (adjusted for A40)
MODEL=$MODEL_CONFIG_DIR_PATH/Qwen-1.5-72B  
GLOBAL_BATCH_SIZE=8 NUM_RETRIEVAL_GPU_PAGE_BLOCKS=200 NUM_STREAMING_GPU_PAGE_BLOCKS=200 \
CHUNK_PREFILL_SIZE=2147000000 \
python qserve_benchmark.py --model $MODEL --benchmarking --precision w4a8kv4 --group-size -1 --kv-quant-granularity fine_grained $common_args
