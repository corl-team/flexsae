# TopK and HierarchicalTopK SAE decoder Triton kernels
# Copyright 2025 T-Tech


from .topk_kernels import triton_topk_sae_loss, topk_sae_loss
from .hierarchical_kernels import triton_hierarchical_sae_loss, hierarchical_sae_loss

__kernel_metadata__ = {
    "license": "Apache-2.0 (with CC-BY-NC-4.0 component; see NOTICE)",
}

__all__ = [
    "__kernel_metadata__",
    "topk_sae_loss",
    "triton_topk_sae_loss",
    "hierarchical_sae_loss",
    "triton_hierarchical_sae_loss",
]
