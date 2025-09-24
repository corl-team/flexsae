# HierarchicalTopK SAE decoder Triton kernels
# Copyright 2025 T-Tech


from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def hierarchical_sae_forward_kernel(
    loss_per_batch_ptr,  # [B]
    final_recon_ptr,  # [B, D]
    indices_ptr,  # [B, K]
    weight_ptr,  # [F, D]
    bias_ptr,  # [D]
    vals_ptr,  # [B, K]
    target_ptr,  # [B, D]
    B: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOOP_NUM_STAGES: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    tl.static_assert((D % BLOCK_D) == 0)
    tl.static_assert((B % BLOCK_B) == 0)
    tl.static_assert((K & (K - 1)) == 0, f"{K=} must be a power of 2")
    tl.static_assert((BLOCK_D & (BLOCK_D - 1)) == 0, f"{BLOCK_D=} must be a power of 2")
    tl.static_assert((BLOCK_B & (BLOCK_B - 1)) == 0, f"{BLOCK_B=} must be a power of 2")

    pid_b = tl.program_id(axis=0).to(tl.int64)
    pid_d = tl.program_id(axis=1).to(tl.int64)

    batch_offsets = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    batch_offsets = batch_offsets.to(tl.int64)
    tl.multiple_of(batch_offsets, BLOCK_B)

    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offset_d = offset_d.to(tl.int64)

    tl.multiple_of(offset_d, BLOCK_D)
    tl.max_contiguous(offset_d, BLOCK_D)

    batch_d_offset = batch_offsets[:, None] * D + offset_d[None, :]

    bias_tile = tl.load(bias_ptr + offset_d).to(tl.float32)

    recon = tl.zeros([BLOCK_B, BLOCK_D], dtype=tl.float32)
    recon += bias_tile[None, :]

    target = tl.load(target_ptr + batch_d_offset).to(tl.float32)

    loss_accum = tl.zeros([BLOCK_B, BLOCK_D], dtype=tl.float32)

    row_idx_ptr = indices_ptr + batch_offsets * K
    row_val_ptr = vals_ptr + batch_offsets * K

    idx = tl.load(row_idx_ptr).to(tl.int64)
    val = tl.load(row_val_ptr).to(tl.float32)
    val = val[:, None]
    weight_tile = tl.load(weight_ptr + idx[:, None] * D + offset_d[None, :]).to(tl.float32)

    for t in tl.range(0, K, num_stages=LOOP_NUM_STAGES):
        recon += weight_tile * val
        diff = recon - target
        loss_accum += diff * diff

        if t + 1 < K:
            idx_next = tl.load(row_idx_ptr + (t + 1)).to(tl.int64)
            val_next = tl.load(row_val_ptr + (t + 1)).to(tl.float32)
            weight_next = tl.load(weight_ptr + idx_next[:, None] * D + offset_d[None, :]).to(tl.float32)

            idx = idx_next
            val = val_next[:, None]
            weight_tile = weight_next

    loss_tile = tl.sum(loss_accum, axis=1)
    tl.atomic_add(
        loss_per_batch_ptr + batch_offsets,
        loss_tile,
        sem="relaxed",
    )
    tl.store(
        final_recon_ptr + batch_d_offset,
        recon,
    )


@triton.jit
def hierarchical_sae_backward_kernel(
    weight_grad_ptr,  # [F, D]
    vals_grad_ptr,  # [B, K]
    bias_grad_ptr,  # [D]
    final_recon_ptr,  # [B, D]
    indices_ptr,  # [B, K]
    weight_ptr,  # [F, D]
    vals_ptr,  # [B, K]
    target_ptr,  # [B, D]
    B: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOOP_NUM_STAGES: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    tl.static_assert((D % BLOCK_D) == 0)
    tl.static_assert((B % BLOCK_B) == 0)
    tl.static_assert((K & (K - 1)) == 0, f"{K=} must be a power of 2")
    tl.static_assert((BLOCK_D & (BLOCK_D - 1)) == 0, f"{BLOCK_D=} must be a power of 2")
    tl.static_assert((BLOCK_B & (BLOCK_B - 1)) == 0, f"{BLOCK_B=} must be a power of 2")

    pid_b = tl.program_id(axis=0).to(tl.int64)
    pid_d = tl.program_id(axis=1).to(tl.int64)

    batch_offsets = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    batch_offsets = batch_offsets.to(tl.int64)
    tl.multiple_of(batch_offsets, BLOCK_B)

    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offset_d = offset_d.to(tl.int64)

    tl.multiple_of(offset_d, BLOCK_D)
    tl.max_contiguous(offset_d, BLOCK_D)

    batch_d_offset = batch_offsets[:, None] * D + offset_d[None, :]

    recon = tl.load(final_recon_ptr + batch_d_offset).to(tl.float32)
    target = tl.load(target_ptr + batch_d_offset).to(tl.float32)

    suffix = tl.zeros([BLOCK_B, BLOCK_D], dtype=tl.float32)
    bias_accum = tl.zeros([BLOCK_B, BLOCK_D], dtype=tl.float32)
    scale = tl.full((), 2.0 / (B * K * D), dtype=tl.float32)

    row_idx_ptr = indices_ptr + batch_offsets * K
    row_val_ptr = vals_ptr + batch_offsets * K
    k_offsets = tl.arange(0, K)
    val_grad_tile = tl.zeros([BLOCK_B, K], dtype=tl.float32)

    step = K - 1
    idx = tl.load(row_idx_ptr + step).to(tl.int64)
    val = tl.load(row_val_ptr + step).to(tl.float32)
    weight_tile = tl.load(weight_ptr + idx[:, None] * D + offset_d[None, :]).to(tl.float32)

    for _ in tl.range(0, K, num_stages=LOOP_NUM_STAGES):
        curr_step = step

        diff = recon - target
        grad_curr = diff * scale
        suffix += grad_curr
        bias_accum += grad_curr

        val_broadcast = val[:, None]
        contrib = suffix * val_broadcast
        tl.atomic_add(
            weight_grad_ptr + idx[:, None] * D + offset_d[None, :],
            contrib,
            sem="relaxed",
        )

        dot_partial = tl.sum(weight_tile * suffix, axis=1)
        mask_curr = k_offsets[None, :] == curr_step
        val_grad_tile = tl.where(mask_curr, dot_partial[:, None], val_grad_tile)

        recon -= weight_tile * val_broadcast

        if curr_step > 0:
            step = curr_step - 1
            idx = tl.load(row_idx_ptr + step).to(tl.int64)
            val = tl.load(row_val_ptr + step).to(tl.float32)
            weight_tile = tl.load(weight_ptr + idx[:, None] * D + offset_d[None, :]).to(tl.float32)

    bias_grad_tile = tl.sum(bias_accum, axis=0)
    tl.atomic_add(
        bias_grad_ptr + offset_d,
        bias_grad_tile,
        sem="relaxed",
    )

    row_val_grad_ptr = vals_grad_ptr + batch_offsets[:, None] * K + k_offsets[None, :]
    tl.atomic_add(
        row_val_grad_ptr,
        val_grad_tile,
        sem="relaxed",
    )


def _hierarchical_sae_forward(
    indices: torch.Tensor,  # [B, K]
    weight: torch.Tensor,  # [F, D]
    vals: torch.Tensor,  # [B, K]
    bias: torch.Tensor,  # [D]
    target: torch.Tensor,  # [B, D]
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, K = indices.shape
    F, D = weight.shape

    loss_per_batch = torch.zeros((B,), dtype=torch.float32, device=weight.device)
    final_recon = torch.empty((B, D), dtype=torch.float32, device=weight.device)

    def _forward_grid(meta):
        return (
            B // meta["BLOCK_B"],
            D // meta["BLOCK_D"],
        )

    hierarchical_sae_forward_kernel[_forward_grid](
        loss_per_batch,
        final_recon,
        indices,
        weight,
        bias,
        vals,
        target,
        B=B,
        D=D,
        K=K,
        BLOCK_D=64,
        LOOP_NUM_STAGES=4,
        BLOCK_B=1,
        num_warps=2,
        num_stages=2,
    )
    loss = loss_per_batch.sum() / (B * K * D)
    return loss, final_recon


def _hierarchical_sae_backward(
    indices: torch.Tensor,  # [B, K]
    weight: torch.Tensor,  # [F, D]
    vals: torch.Tensor,  # [B, K]
    target: torch.Tensor,  # [B, D]
    final_recon: torch.Tensor,  # [B, D]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = weight.device
    B, K = indices.shape
    F, D = weight.shape

    dW = torch.zeros((F, D), dtype=torch.float32, device=device)
    dVals = torch.zeros((B, K), dtype=torch.float32, device=device)
    db = torch.zeros((D,), dtype=torch.float32, device=device)

    def _backward_grid(meta):
        return (
            B // meta["BLOCK_B"],
            D // meta["BLOCK_D"],
        )

    hierarchical_sae_backward_kernel[_backward_grid](
        dW,
        dVals,
        db,
        final_recon,
        indices,
        weight,
        vals,
        target,
        B=B,
        D=D,
        K=K,
        BLOCK_D=32,
        LOOP_NUM_STAGES=16,
        BLOCK_B=16,
        num_warps=8,
        num_stages=8,
    )

    return dW, dVals, db


class HierarchicalSAELossFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        indices: torch.Tensor,  # [B, K]
        weight: torch.Tensor,  # [F, D]
        vals: torch.Tensor,  # [B, K]
        bias: torch.Tensor,  # [D]
        target: torch.Tensor,  # [B, D]
    ):
        loss, final_recon = _hierarchical_sae_forward(indices, weight, vals, bias, target)
        ctx.save_for_backward(indices, weight, vals, target, final_recon)
        return loss

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad):
        indices, weight, vals, target, final_recon = ctx.saved_tensors
        dW, dVals, db = _hierarchical_sae_backward(indices, weight, vals, target, final_recon)

        if grad is not None:
            dW.mul_(grad)
            dVals.mul_(grad)
            db.mul_(grad)

        return None, dW, dVals, db, None


def triton_hierarchical_sae_loss(
    indices: torch.Tensor,  # [B, K]
    weight: torch.Tensor,  # [F, D]
    vals: torch.Tensor,  # [B, K]
    bias: torch.Tensor,  # [D]
    target: torch.Tensor,  # [B, D]
) -> torch.Tensor:
    return HierarchicalSAELossFunction.apply(indices, weight, vals, bias, target)


def hierarchical_sae_loss(
    indices: torch.Tensor,  # [B, K]
    weight: torch.Tensor,  # [F, D]
    vals: torch.Tensor,  # [B, K]
    bias: torch.Tensor,  # [D]
    target: torch.Tensor,  # [B, D]
) -> torch.Tensor:
    emb = weight[indices].to(torch.float32)  # [K, D]
    recon_cum = bias.to(torch.float32) + (emb * vals.unsqueeze(-1)).cumsum(dim=1)
    diff = recon_cum.to(torch.float32) - target.to(torch.float32).unsqueeze(1)
    loss = diff.pow(2).mean()
    return loss
