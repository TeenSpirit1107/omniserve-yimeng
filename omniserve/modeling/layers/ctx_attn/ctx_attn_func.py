import torch
import os
from itertools import count
from block_sparse_attn import (
    token_streaming_attn_func,
    block_streaming_attn_func,
    flash_attn_varlen_func
)

# Environment variable to disable Flash Attention
DISABLE_FLASH_ATTN = os.environ.get("DISABLE_FLASH_ATTN", "0") == "1"

def attention_wrapper(
    q_unpad, k_unpad, v_unpad,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_p=0.0, causal=True,
    head_mask_type = None,
    streaming_info = None,
):
    if head_mask_type is not None and streaming_info is not None:
        attn_output = token_static_sparse_attn(
                        q_unpad, k_unpad, v_unpad,
                        cu_seqlens_q, cu_seqlens_k,
                        max_seqlen_q, max_seqlen_k,
                        head_mask_type, streaming_info
                    )
    else: #dense case
        attn_output = dense_context_attn(
                        q_unpad, k_unpad, v_unpad,
                        cu_seqlens_q, cu_seqlens_k,
                        max_seqlen_q, max_seqlen_k,
                        dropout_p, causal
                    )
    return attn_output
        
def dense_context_attn(
    q_unpad, k_unpad, v_unpad,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_p, causal
):
    if DISABLE_FLASH_ATTN:
        print("Flash Attention disabled, using standard attention")
        return standard_attention_fallback(
            q_unpad, k_unpad, v_unpad,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            dropout_p, causal
        )
    
    try:
        attn_output = flash_attn_varlen_func(
                    q_unpad, k_unpad, v_unpad,
                    cu_seqlens_q, cu_seqlens_k,
                    max_seqlen_q, max_seqlen_k,
                    dropout_p=dropout_p,
                    causal=causal,
                )
        return attn_output
    except RuntimeError as e:
        if "illegal memory access" in str(e) or "CUDA error" in str(e):
            print("Flash Attention failed, falling back to standard attention")
            # Fallback to standard attention
            return standard_attention_fallback(
                q_unpad, k_unpad, v_unpad,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                dropout_p, causal
            )
        else:
            raise e

def standard_attention_fallback(
    q_unpad, k_unpad, v_unpad,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_p, causal
):
    """Fallback to standard attention when Flash Attention fails"""
    # Convert unpad tensors back to padded format for standard attention
    batch_size = len(cu_seqlens_q) - 1
    
    # Reshape tensors for standard attention
    q = q_unpad.view(batch_size, max_seqlen_q, -1)
    k = k_unpad.view(batch_size, max_seqlen_k, -1)
    v = v_unpad.view(batch_size, max_seqlen_k, -1)
    
    # Standard attention computation
    scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
    
    if causal:
        # Create causal mask
        mask = torch.triu(torch.ones(max_seqlen_q, max_seqlen_k, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask.bool(), float('-inf'))
    
    attn_weights = torch.softmax(scores, dim=-1)
    if dropout_p > 0:
        attn_weights = torch.dropout(attn_weights, dropout_p, training=True)
    
    attn_output = torch.matmul(attn_weights, v)
    
    # Convert back to unpad format
    return attn_output.view(-1, attn_output.size(-1))

def block_static_sparse_attn(
    q_unpad, k_unpad, v_unpad,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    head_mask_type, streaming_info
):
    attn_output = block_streaming_attn_func(
                q_unpad, k_unpad, v_unpad,
                cu_seqlens_q, cu_seqlens_k,
                head_mask_type, streaming_info,
                max_seqlen_q, max_seqlen_k,
            )
    return attn_output

def token_static_sparse_attn(
    q_unpad, k_unpad, v_unpad,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    head_mask_type, streaming_info
):
    attn_output = token_streaming_attn_func(
                q_unpad, k_unpad, v_unpad,
                cu_seqlens_q, cu_seqlens_k,
                head_mask_type, streaming_info,
                max_seqlen_q, max_seqlen_k,
            )
    return attn_output