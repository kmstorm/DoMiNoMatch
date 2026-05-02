import torch
import numpy as np
from PIL import Image

@torch.no_grad()
def mixup_one_target(x, y, alpha=1.0, is_bias=False):
    """Returns mixed inputs, mixed targets, and lambda
    """
    # Compute lambda
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if is_bias:
        lam = max(lam, 1 - lam)
    
    # Convert PIL to numpy
    x_np = np.array(x)
    
    # Create mixed image
    mixed_x_np = lam * x_np
    
    # Convert back to PIL
    mixed_x = Image.fromarray(mixed_x_np.astype(np.uint8))
    
    # Mix targets if provided
    if y is not None:
        mixed_y = y  # No mixing needed for single image
    else:
        mixed_y = None
    
    return mixed_x, mixed_y, lam