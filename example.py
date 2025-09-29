# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "kernels",
# ]
# ///

import torch
import numpy as np
from kernels import get_kernel

flex = get_kernel("t-tech/flex-sae")  # Fast Kernels

@torch.compile(fullgraph=True)
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


B = 2048
K = 256
F = 1024 * 128
D = 1024
WARMUP = 5
NUM_ITER = 100
dtype = torch.float32

vals = None
decoder = None
bias = None
target = None
indices = None


def init_parameters():
    global vals, decoder, bias, target, indices
    vals = torch.randn(B, K, dtype=dtype, device="cuda").abs().requires_grad_()
    decoder = torch.randn(F, D, dtype=dtype, device="cuda", requires_grad=True)
    bias = torch.randn(D, dtype=dtype, device="cuda", requires_grad=True)
    target = torch.randn(B, D, dtype=dtype, device="cuda")
    indices = torch.randint(0, F, (B, K), dtype=torch.long, device="cuda")


timing_kernel = []
timing_vanilla = []
torch.cuda.reset_peak_memory_stats()
loss_kernel_list = torch.zeros((100,))
loss_vanilla_list = torch.zeros((100,))


def zero_grad():
    vals.grad = None
    decoder.grad = None
    bias.grad = None
    torch.cuda.empty_cache()


for i in range(NUM_ITER + WARMUP):
    init_parameters()
    start_kernel = torch.cuda.Event(enable_timing=True)
    end_kernel = torch.cuda.Event(enable_timing=True)
    start_vanilla = torch.cuda.Event(enable_timing=True)
    end_vanilla = torch.cuda.Event(enable_timing=True)

    start_kernel.record()
    loss_kernel = flex.triton_hierarchical_sae_loss(indices, decoder, vals, bias, target)
    loss_kernel.backward()
    end_kernel.record()

    zero_grad()
    start_vanilla.record()
    loss_vanilla = hierarchical_sae_loss(indices, decoder, vals, bias, target)
    loss_vanilla.backward()
    end_vanilla.record()
    if i >= WARMUP:
        torch.cuda.synchronize()
        timing_kernel.append(start_kernel.elapsed_time(end_kernel))
        timing_vanilla.append(start_vanilla.elapsed_time(end_vanilla))
        loss_kernel_list[i - WARMUP] = loss_kernel.detach()
        loss_vanilla_list[i - WARMUP] = loss_vanilla.detach()
    zero_grad()

if torch.allclose(loss_kernel, loss_vanilla):
    print("✅ Outputs are close! Everything is good! 🎉")
else:
    print("❌ Outputs mismatch... ⚠️🤔")


print(f"🦎 Triton Kernel Time (Ours): {np.mean(timing_kernel):.4f} ± {np.std(timing_kernel):.4f} ms")
print(f"🔥 Torch Compile Kernel Time: {np.mean(timing_vanilla):.4f} ± {np.std(timing_vanilla):.4f} ms")
print(f"🚀 Speedup: {np.mean(timing_vanilla) / np.mean(timing_kernel):.2f}x")