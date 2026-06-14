"""Utils."""
import numpy as np


def srgb_to_linear(srgb):
    """Convert standard sRGB to Linear RGB."""
    # assumes sRGB values are normalized between 0.0 and 1.0
    srgb = np.asarray(srgb)
    linear_rgb = np.empty_like(srgb)

    linear_mask = srgb <= 0.04045

    linear_rgb[linear_mask] = srgb[linear_mask] / 12.92
    linear_rgb[~linear_mask] = ((srgb[~linear_mask] + 0.055) / 1.055) ** 2.4
    return linear_rgb

    # return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear):
    """Convert Linear RGB back to standard sRGB."""
    linear = np.asarray(linear)
    srgb = np.empty_like(linear)

    mask = linear <= 0.0031308

    srgb[mask] = linear[mask] * 12.92
    srgb[~mask] = 1.055 * (linear[~mask] ** (1.0 / 2.4)) - 0.055
    return np.clip(srgb, 0.0, 1.0)

    # srgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * (linear ** (1.0 / 2.4)) - 0.055)
    # return np.clip(srgb, 0.0, 1.0)


def srgb_to_oklab(rgb_array):
    """Convert a [0 - 255] sRGB array (shape: ..., 3 OR 4) to Oklab space."""
    has_alpha = rgb_array.shape[-1] == 4

    if has_alpha:
        # Separate RGB and Alpha using slicing (creates views, no copy overhead)
        working_rgb = rgb_array[..., :3]
        alpha = rgb_array[..., 3:]
    else:
        working_rgb = rgb_array

    linear_rgb = srgb_to_linear(working_rgb / 255)

    # matrix 1: Linear RGB to LMS cone response space
    m1 = np.array([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005]
    ])
    lms = linear_rgb @ m1.T

    # non-linear cubic root stage
    # avoid negative numbers causing issues in color space boundaries
    lms = np.maximum(lms, 0.0)
    lms_cubic = np.cbrt(lms)

    # matrix 2: Non-linear LMS to Oklab (L, a, b)
    m2 = np.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660]
    ])
    oklab = lms_cubic @ m2.T

    if has_alpha:
        # np.concatenate joins the (L, a, b) array and the (Alpha) array along the last axis
        return np.concatenate([oklab, alpha], axis=-1)

    return oklab


def oklab_to_srgb(oklab_array):
    """
    Convert an Oklab array (shape: ..., 3 OR 4) back to a [0 - 255] sRGB uint8 array.
    Preserves alpha channels if present.
    """
    has_alpha = oklab_array.shape[-1] == 4

    if has_alpha:
        working_oklab = oklab_array[..., :3]
        alpha = oklab_array[..., 3:]
    else:
        working_oklab = oklab_array

    # Inverse Matrix 2: Oklab (L, a, b) to non-linear LMS
    m2_inv = np.array([
        [1.0,  0.3963377774,  0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480]
    ])
    lms_cubic = working_oklab @ m2_inv.T

    # Undo the cubic root stage
    lms = np.power(lms_cubic, 3)

    # Inverse Matrix 1: Linear LMS space to Linear RGB
    m1_inv = np.array([
        [ 4.0767416621, -3.3077115913,  0.2309699292],
        [-1.2684380046,  2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147,  1.7076147010]
    ])
    linear_rgb = lms @ m1_inv.T

    # Convert linear RGB to standard sRGB gamma space
    srgb = linear_to_srgb(linear_rgb)

    # Convert from [0.0, 1.0] float to [0, 255] integer space
    # Clamping is required as some custom Oklab points may sit outside the sRGB gamut display limits
    srgb_255 = np.clip(srgb * 255.0, 0, 255).astype(np.uint8)

    if has_alpha:
        # Reattach alpha component unmodified
        return np.concatenate([srgb_255, alpha.astype(np.uint8)], axis=-1)

    return srgb_255


def oklab_dist(color1, color2, w_L=1, w_a=1, w_b=1):
    """
    Calculate the perceptual distance between two Oklab colors.
    Lightness, a, b are WEIGHTED by parameters w_L, w_a, w_b respectively. 
    Accept arrays of shape (3,) or batches of colors (N, 3).
    """
    color1 = np.asarray(color1)
    color2 = np.asarray(color2)

    delta = color1 - color2
    return np.sqrt((delta[..., 0] * w_L)**2 + (delta[..., 1] * w_a)**2 + (delta[..., 2] * w_b)**2)

    # return np.linalg.norm(color1 - color2, axis=-1)
