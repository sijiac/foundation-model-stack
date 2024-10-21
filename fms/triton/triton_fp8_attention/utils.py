import torch
import triton.language as tl
from typing import Tuple
import numpy as np

MAX_PERCENT_Q = {'P95': None, 'P99': None}
MAX_PERCENT_K = {'P95': None, 'P99': None}
MAX_PERCENT_V = {'P95': None, 'P99': None}

def collect_abs_percentiles(tensor, max_percentiles):
    abs_tensor = torch.abs(tensor)
    quantiles = torch.tensor([0.95, 0.99], device=abs_tensor.device, dtype=torch.float32)
    percentiles = torch.quantile(abs_tensor.to(torch.float32).flatten(), quantiles)

    percentile_dict = {
        'P95': percentiles[0].item(),
        'P99': percentiles[1].item(),
    }

    # Update max_percentiles
    for key in percentile_dict:
        if max_percentiles[key] is None or percentile_dict[key] > max_percentiles[key]:
            max_percentiles[key] = percentile_dict[key]

    return percentile_dict, max_percentiles

def compute_quantile_with_numpy(tensor, quantiles):
    # Convert to NumPy array
    tensor_np = tensor.to(torch.float32).cpu().numpy()
    
    # Compute quantiles using NumPy
    max_abs_val = np.max(np.abs(tensor_np))
    return np.quantile(tensor_np, quantiles), max_abs_val

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
