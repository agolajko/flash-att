import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load
import minimal_attn

# Load your CUDA extension
# minimal_attn = load(
#     name='minimal_attn',
#     sources=['main.cpp', 'flash.cu', 'backward.cu'],
#     extra_cuda_cflags=['-O2'],
#     verbose=True,
#     with_cuda=True,
#     build_directory='./build',
# )


class FlashAttentionFunction(Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        """
        Forward pass - calls your CUDA forward kernel
        Returns O and saves l, m for backward
        """
        # Call your CUDA forward (returns O, l, m)
        O, l, m = minimal_attn.forward(Q, K, V)

        # Save tensors needed for backward pass
        ctx.save_for_backward(Q, K, V, O, l, m)

        return O

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass - calls your CUDA backward kernel
        grad_output is dL/dO
        """
        # Retrieve saved tensors
        Q, K, V, O, l, m = ctx.saved_tensors

        # Call your CUDA backward kernel
        dQ, dK, dV = minimal_attn.backward(Q, K, V, O, grad_output, l, m)

        # Return gradients for each input (Q, K, V)
        # Must return same number of gradients as forward() inputs
        return dQ, dK, dV

# User-friendly wrapper function


def flash_attention(Q, K, V):
    """
    Flash Attention

    Args:
        Q, K, V: Tensors of shape [B, nh, N, d]

    Returns:
        O: Output tensor of shape [B, nh, N, d]
    """
    return FlashAttentionFunction.apply(Q, K, V)
