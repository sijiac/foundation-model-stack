
import torch
import pytest
from typing import Optional, Tuple
import triton.language as tl
import logging
from .quantization import quantize_fp8_per_head

logger = logging.getLogger(__name__)

def get_fp8_constants() -> Tuple[torch.dtype, tl.dtype, float, float]:
    """
    Helper function to get constant values for the current platform.

    Returns:
        pt_dtype (torch.dtype): The correct torch fp8 datatype.
        tl_dtype (tl.dtype): The correct triton fp8 datatype.
        max_fp8 (float): The maximum reprsentable value for the fp8 datatype.
        eps (float): Minimum clip value to prevent divide by zero.
    """
    if torch.version.hip is not None:
        pt_fp8_dtype = torch.float8_e4m3fnuz
        tl_fp8_dtype = tl.float8e4b8
    else:
        pt_fp8_dtype = torch.float8_e4m3fn
        tl_fp8_dtype = tl.float8e4nv
    return pt_fp8_dtype, tl_fp8_dtype, torch.finfo(pt_fp8_dtype).max, 1e-12


# Helper function to generate test tensors
def generate_test_tensor(shape, device="cpu"):
    return torch.randn(shape, device=device)

@pytest.mark.parametrize("shape, device", [
    ((2, 4, 3, 8), "cpu"),
    ((1, 2, 5, 10), "cuda" if torch.cuda.is_available() else "cpu"),
    ((3, 6, 4, 12), "cpu"),
])
def test_quantization_shapes(shape, device):
    # Generate a random tensor with specified shape
    a = generate_test_tensor(shape, device=device)
    
    # Run quantization
    a_fp8, scale = quantize_fp8_per_head(a)
    
    # Check output shapes
    assert a_fp8.shape == a.shape, f"Expected shape {a.shape}, got {a_fp8.shape}"
    assert scale.shape == (shape[0], shape[1]), f"Expected scale shape {(shape[0], shape[1])}, got {scale.shape}"

def test_scaling_range():
    a = generate_test_tensor((2, 3, 4, 8))
    
    # Run quantization
    a_fp8, scale = quantize_fp8_per_head(a)
    
    # Check that scaling is within expected range
    assert torch.all(scale > 0), "Scaling factors must be positive."
    assert torch.all(scale <= 1), "Scaling factors must be less than or equal to 1."

@pytest.mark.parametrize("scale_ub_val", [0.5, 1.0])
def test_clamping_with_scale_ub(scale_ub_val):
    a = generate_test_tensor((2, 3, 4, 8))
    scale_ub = torch.tensor([scale_ub_val])

    # Run quantization with scale upper bound
    a_fp8, scale = quantize_fp8_per_head(a, scale_ub=scale_ub)

    # Ensure that no scaling factor exceeds scale_ub
    assert torch.all(scale <= scale_ub_val), f"Scaling factor exceeded upper bound {scale_ub_val}"

def test_no_inf_or_nan_in_scale():
    a = generate_test_tensor((2, 3, 4, 8))

    # Run quantization
    a_fp8, scale = quantize_fp8_per_head(a)

    # Ensure no NaNs or infinities in scaling
    assert not torch.any(torch.isinf(scale)), "Scale contains infinity values."
    assert not torch.any(torch.isnan(scale)), "Scale contains NaN values."

def test_correct_device_placement():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    a = generate_test_tensor((2, 4, 3, 8), device=device)

    # Run quantization
    a_fp8, scale = quantize_fp8_per_head(a, output_device=torch.device(device))

    # Check if tensors are placed on the correct device
    assert a_fp8.device == torch.device(device), f"Output tensor is not on the correct device. Expected {torch.device(device)}, got {a_fp8.device}."
    assert scale.device == torch.device(device), f"Scale tensor is not on the correct device. Expected {torch.device(device)}, got {scale.device}."
