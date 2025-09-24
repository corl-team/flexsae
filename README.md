# Flex SAE Kernels

[![ArXiv](https://img.shields.io/badge/arXiv-2505.24473-b31b1b.svg)](https://arxiv.org/abs/2505.24473)

Fused Triton implementations of the TopK and HierarchicalTopK sparse autoencoder (SAE) decoder losses described in *Train One Sparse Autoencoder Across Multiple Sparsity Budgets to Preserve Interpretability and Accuracy*.

**This work has been accepted to [EMNLP 2025](https://2025.emnlp.org/).**

## Overview
- `torch-ext/flex_sae/` contains the Triton kernels alongside torch reference implementations.
- `tests/` hosts CUDA-backed property tests that ensure numerical parity across dtypes and kernels.
- `build.toml`, `flake.nix` integrate the project with [Hugging Face kernel-builder](https://github.com/huggingface/kernel-builder).

The Triton kernels target CUDA GPUs and focus on reducing the latency gap between TopK and HierarchicalTopK decoders while keeping memory usage flat.

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
