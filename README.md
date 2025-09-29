# Flex SAE Kernels

[![ArXiv](https://img.shields.io/badge/arXiv-2505.24473-b31b1b.svg)](https://arxiv.org/abs/2505.24473)

Fused Triton implementations of the TopK and HierarchicalTopK sparse autoencoder (SAE) decoder losses described in *Train One Sparse Autoencoder Across Multiple Sparsity Budgets to Preserve Interpretability and Accuracy*.

**This work has been accepted to [EMNLP 2025](https://2025.emnlp.org/).**

## What is released?

 - Fast TopK kernel for SAE (slightly modified version from xformers) `torch-ext/flex_sae/topk_kernels.py`
 - Fast HierarchicalTopK kernels (see our [paper](https://arxiv.org/abs/2505.24473)) `torch-ext/flex_sae/hierarchical_kernels.py`.


## Quickstart

Kernels are available via loading from hub, they have the following signature:
```python
from kernels import get_kernel


flex = get_kernel('t-tech/flex-sae')

top_k_kernel = flex.triton_topk_sae_loss
hierarchical_top_k_kernel = flex.triton_hierarchical_sae_loss

"B -- batch size, K -- top-k, F -- dictionary size, D -- model hidden dim"

loss: torch.Tensor = top_k_kernel(
    indices: torch.Tensor,  # [B, K]
    weight: torch.Tensor,  # [F, D]
    vals: torch.Tensor,  # [B, K]
    bias: torch.Tensor,  # [D]
    target: torch.Tensor,  # [B, D]
)

loss: torch.Tensor = hierarchical_top_k_kernel(
    indices: torch.Tensor,  # [B, K]
    weight: torch.Tensor,  # [F, D]
    vals: torch.Tensor,  # [B, K]
    bias: torch.Tensor,  # [D]
    target: torch.Tensor,  # [B, D]
)
```

## Overview
- `torch-ext/flex_sae/` contains the Triton kernels alongside torch reference implementations.
- `tests/` hosts CUDA-backed property tests that ensure numerical parity across dtypes and kernels.
- `build.toml`, `flake.nix` integrate the project with [Hugging Face kernel-builder](https://github.com/huggingface/kernel-builder).

The Triton kernels target CUDA GPUs and focus on reducing the latency gap between TopK and HierarchicalTopK decoders while keeping memory usage flat.

## Example

You can find example usage in [example.py](example.py). 
```python
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

flex = get_kernel("t-tech/flex-sae")  #Fast Kernels

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
        loss_kernel_list[i-warmup] = loss_kernel.detach()
        loss_vanilla_list[i-warmup] = loss_vanilla.detach()
    zero_grad()

if torch.allclose(loss_kernel, loss_vanilla):
    print("✅ Outputs are close! Everything is good! 🎉")
else:
    print("❌ Outputs mismatch... ⚠️🤔")


print(f"🦎 Triton Kernel Time (Ours): {np.mean(timing_kernel):.4f} ± {np.std(timing_kernel):.4f} ms")
print(f"🔥 Torch Compile Kernel Time: {np.mean(timing_vanilla):.4f} ± {np.std(timing_vanilla):.4f} ms")
print(f"🚀 Speedup: {np.mean(timing_vanilla) / np.mean(timing_kernel):.2f}x")
```

Run it with `uv run https://huggingface.co/t-tech/flex-sae/resolve/main/example.py`.

## Performance
Benchmarks were collected on a workload with dictionary size $F = 65 536$, embedding dimension $D = 2304$, and sparsity budgets $K \in \{32, 64, 128\}$. Latency is reported as time per training step (milliseconds) and memory as peak device usage (GiB).

| Decoder backend | K=32 (ms / GiB) | K=64 (ms / GiB) | K=128 (ms / GiB) |
| --- | --- | --- | --- |
| **Pure torch-compiled** | | | |
| TopK | 8.787 / 2.92 | 11.746 / 2.92 | 18.877 / 2.93 |
| HierarchicalTopK | 12.824 / 6.29 | 23.379 / 10.79 | 43.851 / 19.80 |
| **Triton kernels** | | | |
| TopK | 5.576 / 2.92 | 6.339 / 2.92 | 7.961 / 2.93 |
| HierarchicalTopK | **6.696 / 2.92** | **7.995 / 2.92** | **10.609 / 2.93** |

Across the evaluated sparsity budgets the fused Triton HierarchicalTopK kernel matches TopK kernels on memory use while remaining consistently faster than the reference torch implementation.

## License & Attribution
- All files except `torch-ext/flex_sae/topk_kernels.py` are released under the [Apache License 2.0](LICENSE).
- `torch-ext/flex_sae/topk_kernels.py` includes code adapted from Facebook Research's [memory](https://github.com/facebookresearch/memory) project, originally published under the Creative Commons Attribution-NonCommercial 4.0 International License. That component therefore remains available for non-commercial use only; see [NOTICE](NOTICE) for details.

## Citation
```bibtex
@misc{balagansky2025trainsparseautoencodermultiple,
      title={Train One Sparse Autoencoder Across Multiple Sparsity Budgets to Preserve Interpretability and Accuracy},
      author={Nikita Balagansky and Yaroslav Aksenov and Daniil Laptev and Vadim Kurochkin and Gleb Gerasimov and Nikita Koryagin and Daniil Gavrilov},
      year={2025},
      eprint={2505.24473},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.24473},
}
```
