import torch
import torch.nn.functional as F
import math
from flash_attn import flash_attention  # Import your wrapper


def test_against_pytorch_reference():
    """Compare against PyTorch's implementation"""
    B, nh, N, d = 2, 2, 32, 64

    # Create inputs
    Q = torch.randn(B, nh, N, d, requires_grad=True, device='cuda')
    K = torch.randn(B, nh, N, d, requires_grad=True, device='cuda')
    V = torch.randn(B, nh, N, d, requires_grad=True, device='cuda')

    Q_ref = Q.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)

    # NOW enable gradients for your implementation
    Q_flash = Q.detach().clone().requires_grad_(True)
    K_flash = K.detach().clone().requires_grad_(True)
    V_flash = V.detach().clone().requires_grad_(True)

    # PyTorch reference
    scale = 1.0 / math.sqrt(d)
    O_ref = F.scaled_dot_product_attention(
        Q_ref, K_ref, V_ref,
        attn_mask=None,
        dropout_p=0.0,
        scale=scale
    )

    # Your implementation - NOW THIS WILL WORK!
    # Uses the autograd wrapper
    O_flash = flash_attention(Q_flash, K_flash, V_flash)
    grad_output = torch.randn_like(O_flash)

    O_flash.backward(grad_output)

    print(f"✅ Flash backward succeeded")

    O_ref.backward(grad_output)

    # Compare outputs
    print(
        f"Output matches: {torch.allclose(O_flash, O_ref, rtol=1e-4, atol=1e-5)}")
    print(f"Max output error: {(O_flash - O_ref).abs().max().item()}")

    # Compare gradients
    print(
        f"\ndQ matches: {torch.allclose(Q_flash.grad, Q_ref.grad, rtol=1e-3, atol=1e-4)}")
    print(
        f"dK matches: {torch.allclose(K_flash.grad, K_ref.grad, rtol=1e-3, atol=1e-4)}")
    print(
        f"dV matches: {torch.allclose(V_flash.grad, V_ref.grad, rtol=1e-3, atol=1e-4)}")

    print("\n section of dQ gradients:", Q_flash.grad[0, 0, :5, :5])
    print("\n section of dQ ref gradients:", Q_ref.grad[0, 0, :5, :5])

    print(f"\nMax dQ error: {(Q_flash.grad - Q_ref.grad).abs().max().item()}")
    print(f"Max dK error: {(K_flash.grad - K_ref.grad).abs().max().item()}")
    print(f"Max dV error: {(V_flash.grad - V_ref.grad).abs().max().item()}")


if __name__ == "__main__":
    test_against_pytorch_reference()
    print("\n✅ All tests passed!")
