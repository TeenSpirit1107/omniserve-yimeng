#!/bin/bash

# LServe launch script for A40 GPU (adjusted for 46GB memory)
export CUDA_VISIBLE_DEVICES=1

# Use existing attention patterns
model_name=Llama-3-8B-Instruct-Gradient-1048k
model_path=/local/ymteng/models/Llama-3-8B-Instruct-Gradient-1048k

batch_size=1
prefill_len_list=(1000 2000 4000 8000 16000)
decode_len=128
precision=w8a8kv8
kv_quant_granularity=per_tensor

static_sparsity=0.5
sparse_prefill_mode=1
sparse_decode_mode=1
dynamic_attn_budget=2048
dynamic_select_interval=4
sub_chunk_per_block=4

# Download the model config files for benchmarking
MODEL_CONFIG_DIR_PATH=./QServe-benchmarks
if [ ! -d "$MODEL_CONFIG_DIR_PATH" ]; then
    git clone https://huggingface.co/datasets/mit-han-lab/QServe-benchmarks
fi

for prefill_len in "${prefill_len_list[@]}"; do
    bash scripts/lserve_benchmark/benchmark.sh \
        $model_path \
        attn_patterns/$model_name \
        $batch_size $prefill_len $decode_len \
        $precision $kv_quant_granularity \
        $static_sparsity $sparse_prefill_mode \
        $sparse_decode_mode $dynamic_attn_budget $dynamic_select_interval $sub_chunk_per_block
done 