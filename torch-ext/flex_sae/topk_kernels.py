# TopK SAE decoder Triton kernels
# Copyright 2025 T-Tech
# This code is adapted from Facebook Research under the
# Creative Commons Attribution-NonCommercial 4.0 International License.
# Original code can be found at: https://github.com/facebookresearch/memory


from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def embedding_bag_forward_kernel(
    out_ptr,  # [B, D]
    indices_ptr,  # [B, K]
    weight_ptr,  # [F, D]
    vals_ptr,  # [B, K]
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tl.static_assert((D % BLOCK_D) == 0)
    tl.static_assert((K & (K - 1)) == 0, f"{K=} must be a power of 2")
    tl.static_assert((BLOCK_D & (BLOCK_D - 1)) == 0, f"{BLOCK_D=} must be a power of 2")

    b = tl.program_id(axis=0).to(tl.int64)
    pid_d = tl.program_id(axis=1).to(tl.int64)

    off_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    out_value = tl.zeros([BLOCK_D], dtype=tl.float32)
    for i in tl.range(K):
        my_index = tl.load(indices_ptr + b * K + i).to(tl.int64)
        my_scaling = tl.load(vals_ptr + b * K + i)
        w_tile = tl.load(weight_ptr + my_index * D + off_d).to(tl.float32)
        out_value += w_tile * my_scaling

    tl.store(out_ptr + b * D + off_d, out_value)


def embedding_bag_forward(
    indices: torch.Tensor,  # [B, K]
    weight: torch.Tensor,  # [F, D]
    vals: torch.Tensor,  # [B, K]
) -> torch.Tensor:
    B, K = indices.shape
    D = weight.shape[1]

    trt_out = torch.empty([B, D], dtype=weight.dtype, device=weight.device)

    def _forward_grid(meta):
        return (B, D // meta["BLOCK_D"])

    embedding_bag_forward_kernel[_forward_grid](
        trt_out,
        indices,
        weight,
        vals,
        D=D,
        K=K,
        BLOCK_D=64,
        num_warps=1,
        num_stages=1,
    )
    return trt_out


@triton.jit
def count_per_embedding_kernel(
    count_per_emb_ptr,  # [F + 1]
    indices_ptr,  # [B, K]
    K: tl.constexpr,
):
    batch_id = tl.program_id(axis=0).to(tl.int64)
    for t in tl.range(K):
        embedding_id = tl.load(indices_ptr + batch_id * K + t)
        tl.atomic_add(count_per_emb_ptr + embedding_id + 1, 1, sem="relaxed")


@triton.jit
def map_embeddings_and_outputs_kernel(
    reverse_mapping_ptr,  # [B * K]
    mapping_write_pos_ptr,  # [F]
    indices_ptr,  # [B, K]
    K: tl.constexpr,
):
    batch_id = tl.program_id(axis=0).to(tl.int64)
    for t in tl.range(K):
        embedding_id = tl.load(indices_ptr + batch_id * K + t)
        write_pos = tl.atomic_add(mapping_write_pos_ptr + embedding_id, 1, sem="relaxed")
        tl.store(reverse_mapping_ptr + write_pos, batch_id * K + t)


@triton.jit
def aggregate_gradient_for_embedding_kernel(
    weight_grad_ptr,  # [F, D]
    vals_grad_ptr,  # [B, K]
    weight_ptr,  # [F, D]
    emb_begin_pos_ptr,  # [F + 1]
    reverse_mapping_ptr,  # [B * K]
    vals_ptr,  # [B, K]
    gradient_ptr,  # [B, D]
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tl.static_assert((D % BLOCK_D) == 0)
    tl.static_assert((K & (K - 1)) == 0, f"{K=} must be a power of 2")
    tl.static_assert((BLOCK_D & (BLOCK_D - 1)) == 0, f"{BLOCK_D=} must be a power of 2")

    e = tl.program_id(axis=0).to(tl.int64)
    pid_d = tl.program_id(axis=1).to(tl.int64)

    off_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    begin = tl.load(emb_begin_pos_ptr + e)
    end = tl.load(emb_begin_pos_ptr + e + 1)

    w_row_tile = tl.load(weight_ptr + e * D + off_d).to(tl.float32)
    w_grad_tile = tl.zeros([BLOCK_D], dtype=tl.float32)

    for idx in tl.range(begin, end):
        out_linear = tl.load(reverse_mapping_ptr + idx).to(tl.int64)
        b = out_linear // K

        psw = tl.load(vals_ptr + out_linear)
        g_tile = tl.load(gradient_ptr + b * D + off_d).to(tl.float32)

        w_grad_tile += psw * g_tile

        psw_grad_partial = tl.sum(g_tile * w_row_tile)
        tl.atomic_add(vals_grad_ptr + out_linear, psw_grad_partial, sem="relaxed")

    tl.store(weight_grad_ptr + e * D + off_d, w_grad_tile)


def embedding_bag_backward(
    indices: torch.Tensor,  # [B, K]
    weight: torch.Tensor,  # [F, D]
    vals: torch.Tensor,  # [B, K]
    gradient: torch.Tensor,  # [B, D]
) -> Tuple[torch.Tensor, torch.Tensor]:
    F, D = weight.shape
    B, K = indices.shape

    count_per_emb = torch.zeros((F + 1,), dtype=torch.uint32, device=indices.device)
    count_per_embedding_kernel[(B,)](count_per_emb, indices, K=K, num_warps=1)

    emb_begin_pos = count_per_emb.cumsum(0)  # [F + 1]

    reverse_mapping = torch.empty([B * K], dtype=torch.uint32, device=indices.device)
    assert B * K <= 2 ** (reverse_mapping.dtype.itemsize * 8) - 1

    map_embeddings_and_outputs_kernel[(B,)](
        reverse_mapping_ptr=reverse_mapping,
        mapping_write_pos_ptr=emb_begin_pos.clone(),
        indices_ptr=indices,
        K=K,
        num_warps=1,
    )

    weight_grad = torch.empty_like(weight, dtype=torch.float32)  # [F, D]
    vals_grad = torch.zeros_like(vals, dtype=torch.float32)  # [B, K]

    def _forward_grid(meta):
        return (F, D // meta["BLOCK_D"])

    aggregate_gradient_for_embedding_kernel[_forward_grid](
        weight_grad_ptr=weight_grad,
        vals_grad_ptr=vals_grad,
        weight_ptr=weight,
        emb_begin_pos_ptr=emb_begin_pos,
        reverse_mapping_ptr=reverse_mapping,
        vals_ptr=vals,
        gradient_ptr=gradient,
        D=D,
        K=K,
        BLOCK_D=256,
        num_warps=1,
        num_stages=2,
    )
    return weight_grad, vals_grad


class xFormersEmbeddingBag(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        indices: torch.Tensor,  # [B, K]
        weight: torch.Tensor,  # [F, D]
        vals: torch.Tensor,  # [B, K]
    ) -> torch.Tensor:
        ctx.save_for_backward(indices, weight, vals)
        return embedding_bag_forward(indices, weight, vals)  # [B, D]

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, gradient):
        indices, weight, vals = ctx.saved_tensors
        weight_g, vals_g = embedding_bag_backward(
            indices,
            weight,
            vals,
            gradient,
        )
        return None, weight_g, vals_g


def triton_topk_sae_loss(
    indices: torch.Tensor,  # [B, K]
    weight: torch.Tensor,  # [F, D]
    vals: torch.Tensor,  # [B, K]
    bias: torch.Tensor,  # [D]
    target: torch.Tensor,  # [B, D]
) -> torch.Tensor:
    recon = bias.to(torch.float32) + xFormersEmbeddingBag.apply(indices, weight, vals)
    diff = recon.to(torch.float32) - target.to(torch.float32)
    loss = diff.pow(2).mean()
    return loss


def topk_sae_loss(
    indices: torch.Tensor,  # [B, K]
    weight: torch.Tensor,  # [F, D]
    vals: torch.Tensor,  # [B, K]
    bias: torch.Tensor,  # [D]
    target: torch.Tensor,  # [B, D]
) -> torch.Tensor:
    emb = weight[indices].to(torch.float32)  # [K, D]
    recon = bias.to(torch.float32) + (emb * vals.unsqueeze(-1)).sum(dim=1)
    diff = recon.to(torch.float32) - target.to(torch.float32)
    loss = diff.pow(2).mean()
    return loss
