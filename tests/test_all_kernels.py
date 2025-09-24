from typing import Callable
import pytest
import torch

pytest.importorskip("torch.cuda")
from .test_setup import DTYPES, DTYPE_TO_TOLS, PARAMS, SEED
from flex_sae import (
    triton_hierarchical_sae_loss,
    hierarchical_sae_loss,
    triton_topk_sae_loss,
    topk_sae_loss,
)


@pytest.fixture(autouse=True)
def _set_cuda_default_device():
    torch.set_default_device("cuda")


def run_funcs(B, K, F, D, dtype, *, kernel_foo: Callable, ref_foo: Callable):
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 not supported on this GPU")

    torch.manual_seed(SEED)

    indices = torch.randint(0, F, (B, K), dtype=torch.long, device="cuda")

    vals = torch.randn(B, K, dtype=dtype, device="cuda").abs().requires_grad_()
    decoder = torch.randn(F, D, dtype=dtype, device="cuda", requires_grad=True)
    bias = torch.randn(D, dtype=dtype, device="cuda", requires_grad=True)
    target = torch.randn(B, D, dtype=dtype, device="cuda")

    sv_ref = vals.clone().detach().requires_grad_()
    dec_ref = decoder.clone().detach().requires_grad_()
    bias_ref = bias.clone().detach().requires_grad_()

    loss_f = kernel_foo(indices, decoder, vals, bias, target)
    loss_r = ref_foo(indices, dec_ref, sv_ref, bias_ref, target)

    torch.testing.assert_close(loss_f, loss_r, **DTYPE_TO_TOLS[dtype])

    grad_out = torch.randn((), device="cuda", dtype=torch.float32)
    loss_f.backward(grad_out)
    loss_r.backward(grad_out.clone())

    torch.testing.assert_close(vals.grad, sv_ref.grad, **DTYPE_TO_TOLS[dtype])
    torch.testing.assert_close(decoder.grad, dec_ref.grad, **DTYPE_TO_TOLS[dtype])
    torch.testing.assert_close(bias.grad, bias_ref.grad, **DTYPE_TO_TOLS[dtype])

    assert indices.grad is None


@pytest.mark.parametrize("B, K, F, D", PARAMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_triton_hierarchical_sae_loss_and_grads(B, K, F, D, dtype):
    run_funcs(B, K, F, D, dtype, kernel_foo=triton_hierarchical_sae_loss, ref_foo=hierarchical_sae_loss)
    torch.cuda.empty_cache()


@pytest.mark.parametrize("B, K, F, D", PARAMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_topk_sae_loss_and_grads(B, K, F, D, dtype):
    run_funcs(
        B, K, F, D, dtype, kernel_foo=triton_topk_sae_loss, ref_foo=topk_sae_loss
    )
    torch.cuda.empty_cache()
