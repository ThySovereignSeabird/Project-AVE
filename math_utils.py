import numpy as np

def weighted_mean_and_std(values, weights):
    v = np.array(values)
    w = np.array(weights)

    mean = np.average(v, weights=w)
    variance = np.average((v - mean) ** 2, weights=w)
    return mean, np.sqrt(variance)

def srgb_to_linear_rgb(channel):
    """
    channel must be in a float in [0, 1]
    """
    return np.where(channel <= 0.04045,
                    channel / 12.92,
                    ((channel + 0.055) / (1.055)) ** 2.4)

def rgb_to_luminance(rgb_array):
    """
    Compute WCAG-compliant relative luminance from an RGB array.
    Input: rgb_array of shape (H, W, 3), dtype float in [0, 1] or uint8 in [0, 255]
    Output: luminance array (H, W) in [0, 1]
    """
    # Normalize 0-255 to 0-1 
    if rgb_array.dtype == np.uint8:
        rgb_array = rgb_array / 255.0

    # Linearize each channel
    r_lin = srgb_to_linear_rgb(rgb_array[..., 0])
    g_lin = srgb_to_linear_rgb(rgb_array[..., 1])
    b_lin = srgb_to_linear_rgb(rgb_array[..., 2])

    # Apply WCAG relative luminance formula (Rec. 709)
    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

def get_contrast_ratio(lum1, lum2):
    L1, L2 = max(lum1, lum2), min(lum1, lum2)
    return (L1 + 0.05) / (L2 + 0.05)
