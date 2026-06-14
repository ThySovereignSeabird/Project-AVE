import numpy as np

from skimage.color import rgb2hsv
from skimage.filters import sobel
from skimage.measure import label, regionprops
from scipy import ndimage

from old.math_utils import weighted_mean_and_std, rgb_to_luminance, get_contrast_ratio

LIGHT_BACKGROUND_LUM = 0.2582 # luminance of #8b8b8b
DARK_BACKGROUND_LUM = 0.0437 # luminance of #3b3b3b
# print(rgb_to_luminance(np.array((139, 139, 139), dtype=np.uint8)))
# print(rgb_to_luminance(np.array((59, 59, 59), dtype=np.uint8)))

FOUR_CONTIGUITY = np.array([
        [False, True, False],
        [True,  True, True],
        [False, True, False]
    ], dtype=bool)
TOP_CONTIGUITY = np.array([
        [False, True, False],
        [False,  True, False],
        [False, False, False]
    ], dtype=bool)
BOTTOM_CONTIGUITY = np.array([
        [False, False, False],
        [False,  True, False],
        [False, True, False]
    ], dtype=bool)

def get_palette_contrast():
    """
    Get the contrast metrics of a palette.
    """
    pass

def get_texture_contrast(img):
    """
    Input an image to get a dict containing the contrast metrics of its texture.
    """
    img_arr = np.array(img)
    rgb = img_arr[..., :3]
    alpha = img_arr[..., 3]
    opaque_mask = alpha == 255
    inner_mask = get_inner_mask(opaque_mask)
    outline_mask = get_outline_mask(opaque_mask)
    top_mask, bottom_mask = get_top_and_bottom_masks(opaque_mask)
    
    area = np.count_nonzero(opaque_mask)
    area_outline = int(np.sum(outline_mask))

    lums = rgb_to_luminance(rgb)
    contrast_values, contrast_weights = get_internal_contrast(lums, opaque_mask)
    contrast_avg, contrast_dev = weighted_mean_and_std(contrast_values, contrast_weights)
    lums_inner = lums[inner_mask]
    lums_outline = lums[outline_mask]
    lums_opaque = lums[opaque_mask]

    luminance_avg = np.mean(lums_opaque)
    luminance_median = np.median(lums_opaque)
    lum_outline_avg = np.mean(lums_outline)

    lums_top = lums[top_mask]
    lums_bottom = lums[bottom_mask]
    lum_top_avg = np.mean(lums_top)
    lum_bottom_avg = np.mean(lums_bottom)

    unique_lums = np.unique(lums_opaque)
    luminance_highlight = ((unique_lums[-1] + unique_lums[-2]) / 2) if len(unique_lums) > 2 else luminance_avg
    
    return {
        "area": area, # Number of opaque pixels
        "area_outline": area_outline, # Number of outline pixels
        "area_inner": area - area_outline, # Number of inner pixels
        "color_count": get_color_count(rgb[opaque_mask]), # Number of distinct opaque color codes
        # Average edge strength within the inner pixels
        "internal_edge_density": float(get_internal_edge_density(lums, inner_mask)),
        # Internal contrast ratio = WCAG contrast ratios between adjacent opaque pixels
        "internal_contrast_ratio_avg": float(contrast_avg),
        "internal_contrast_ratio_dev": float(contrast_dev),
        "luminance_avg": float(luminance_avg), # Average luminance
        "luminance_dev": float(np.std(lums_opaque)), # Deviation of luminance
        "luminance_median": float(luminance_median), # Median of luminance
        "lum_inner_avg": float(np.mean(lums_inner)),
        "lum_inner_dev": float(np.std(lums_inner)),
        "lum_outline_avg": float(lum_outline_avg),
        "lum_outline_dev": float(np.std(lums_outline)),
        # Outline boldness = average contrast ratio between each outline pixel and its neighboring non-outline pixels
        "outline_boldness": float(np.mean(get_outline_boldness(lums, inner_mask, outline_mask))),
        # Highlight luminance = average of the two largest distinct luminances
        "luminance_highlight": float(luminance_highlight),
        # Various contrast ratios targeting specific tones of the texture
        "contrast_highlight_outline": float(get_contrast_ratio(luminance_highlight, lum_outline_avg)),
        "contrast_highlight_median": float(get_contrast_ratio(luminance_highlight, luminance_median)),
        "contrast_median_outline": float(get_contrast_ratio(luminance_median, lum_bottom_avg)),
        "contrast_top_bottom": float(get_contrast_ratio(lum_top_avg, lum_bottom_avg))
    }

def get_texture_contrast_old(img, name):
    """
    Here for reference
    """
    img_arr = np.array(img)
    rgb = img_arr[..., :3]
    alpha = img_arr[..., 3]
    opaque_mask = alpha == 255
    opaque_rgb = rgb[opaque_mask] # lacks positional info

    area = np.count_nonzero(opaque_mask)
    area_outline = int(np.sum(get_outline_mask(opaque_mask)))
    color_count = get_color_count(opaque_rgb)
    color_4contiguity, color_8contiguity = get_color_contiguity(rgb, opaque_mask)
    edge_density = np.mean(sobel(rgb.mean(axis=2)))

    lum = rgb_to_luminance(rgb)
    contrast_values, contrast_weights = get_internal_contrast(lum, opaque_mask)
    contrast_avg, contrast_dev = weighted_mean_and_std(contrast_values, contrast_weights)
    inner_lums, outline_lums, pop_inner_light, pop_outline_light, pop_inner_dark, pop_outline_dark = get_pop_factors(lum, opaque_mask)

    return {
        "name": name, # Namespace and title of texture
        "area": area, # The number of opaque pixels in the texture 
        "area_outline": area_outline, # The number of outline pixels in the texture
        "area_inner": area - area_outline, # The number of inner pixels in the texture
        "color_count": color_count, # The number of distinct opaque color codes in the texture 
        # Color contiguity = The pixel counts of each contiguous bloc of color in the texture
        "color_4contiguity_avg": float(np.mean(color_4contiguity)),
        "color_4contiguity_dev": float(np.std(color_4contiguity)),
        "color_8contiguity_avg": float(np.mean(color_8contiguity)),
        "color_8contiguity_dev": float(np.std(color_8contiguity)),
        "edge_density": float(edge_density), # Grayscales, then applies Sobel filter to find the texture's average edge strength
        # Internal contrast ratio = WCAG contrast ratios between adjacent opaque pixels in the texture
        "internal_contrast_ratio_avg": float(contrast_avg),
        "internal_contrast_ratio_dev": float(contrast_dev),
        "luminance_avg": float(np.mean(lum)), # Average luminance
        "luminance_dev": float(np.std(lum)), # Deviation of luminance
        "lum_inner_avg": float(np.mean(inner_lums)),
        "lum_inner_dev": float(np.std(inner_lums)),
        "lum_outline_avg": float(np.mean(outline_lums)),
        "lum_outline_dev": float(np.std(outline_lums)),
        # Pop factor = WCAG contrast ratios between the specified region of the texture and a neutral background
        "pop_inner_light_avg": float(np.mean(pop_inner_light)),
        "pop_inner_light_dev": float(np.std(pop_inner_light)),
        "pop_outline_light_avg": float(np.mean(pop_outline_light)),
        "pop_outline_light_dev": float(np.std(pop_outline_light)),
        "pop_inner_dark_avg": float(np.mean(pop_inner_dark)),
        "pop_inner_dark_dev": float(np.std(pop_inner_dark)),
        "pop_outline_dark_avg": float(np.mean(pop_outline_dark)),
        "pop_outline_dark_dev": float(np.std(pop_outline_dark)),
    }

def get_touches_transparent_mask(opaque_mask):
    """
    Input opaque mask to get all transparent and outline pixels.
    """
    inner_mask = ndimage.binary_erosion(
        opaque_mask,
        structure=FOUR_CONTIGUITY,
        border_value=False # Treat outside image as transparent for erosion
    )

    return ~inner_mask

def get_inner_mask(opaque_mask):
    """
    Input opaque mask to get all inner pixels, OR the outline mask if no inner pixels are found.
    """
    transparent_mask = get_touches_transparent_mask(opaque_mask)
    inner_mask = ~transparent_mask
    if not inner_mask.any():
        return opaque_mask & transparent_mask
    else:
        return inner_mask

def get_outline_mask(opaque_mask):
    """
    Input opaque mask to get all outline pixels.
    """
    return opaque_mask & get_touches_transparent_mask(opaque_mask)

def get_top_and_bottom_masks(opaque_mask):
    """
    Input opaque mask to get top outline and bottom outline.
    """
    transparent_mask = ~opaque_mask

    top_outline = ndimage.binary_erosion(
        opaque_mask,
        structure=TOP_CONTIGUITY,
        border_value=False # Treat outside image as transparent for erosion
    ) & opaque_mask
    bottom_outline = ndimage.binary_erosion(
        opaque_mask,
        structure=BOTTOM_CONTIGUITY,
        border_value=False # Treat outside image as transparent for erosion
    ) & opaque_mask

    return top_outline, bottom_outline

def get_color_count(opaque_rgb):
    """
    Input rgb with transparent pixels removed to count the number of distinct colors.
    """
    return np.unique(opaque_rgb.reshape(-1, 3), axis=0).shape[0]

def get_color_contiguity(rgb, opaque_mask):
    """
    Input an rgb image array (0-255).
    Returns bloc_sizes4 and bloc_sizes8: list of ints, each the area of a contiguous same-color opaque region.
    Difference is the flag for connectivity (1 = 4-adjacency, 2 = 8-adjacency)
    """
    H, W, C = rgb.shape
    assert C == 3, "Input image must have 3 channels (RGB)"
    assert opaque_mask.shape == (H, W), "Opaque mask must have the same H, W as the RGB image"
    assert opaque_mask.dtype == bool, "Opaque mask must be a bool array"

    # 2D array of colors, each represented as an int
    rgb_int = rgb[:, :, 0].astype(np.int64) * (256 * 256) + \
              rgb[:, :, 1].astype(np.int64) * 256 + \
              rgb[:, :, 2].astype(np.int64)
    rgb_int[~opaque_mask] = -1 # Sets transparent pixels to invalid RGB integer
    
    unique_colors = np.unique(rgb_int[opaque_mask])

    bloc_sizes4 = []
    bloc_sizes8 = []
    for color in unique_colors:
        # Label connected components of the masked color
        color_mask = (rgb_int == color)
        labeled_array4 = label(color_mask, connectivity=1)
        labeled_array8 = label(color_mask, connectivity=2)
        
        # Record area of each connected component
        for region in regionprops(labeled_array4):
            bloc_sizes4.append(region.area)
        for region in regionprops(labeled_array8):
            bloc_sizes8.append(region.area)
    
    return bloc_sizes4, bloc_sizes8

def get_internal_contrast(lum, opaque_mask):
    """
    Input a luminance array and a mask to get data (value array, weight array) on the internal contrast.
    """
    H, W = lum.shape

    adjacents = [(0, 1), (1, 0)]  # right, down
    diagonals = [(1, 1), (1, -1)] # downright, downleft

    contrasts = []
    weights = []
    for y in range(H):
        for x in range(W):
            if not opaque_mask[y, x]:
                continue  # skip transparent / unmasked pixel

            for directions, weight in [(adjacents, 1), (diagonals, 0.707)]:
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and opaque_mask[ny, nx]:
                        contrast_ratio = get_contrast_ratio(lum[y, x], lum[ny, nx])
                        contrasts.append(contrast_ratio)
                        weights.append(weight)

    if len(contrasts) == 0:
        return None, None

    return np.array(contrasts), np.array(weights)

def get_internal_edge_density(lums, inner_mask):
    """
    Calculates average Sobel edge magnitude for inner pixels only.
    """
    # Apply Sobel filter in x and y directions
    #    mode='constant', cval=0.0: Pads the image with zeros at the borders,
    #    which is generally suitable for textures where outside is transparent.
    sobel_x = sobel(lums, axis=1, mode='constant', cval=0.0)
    sobel_y = sobel(lums, axis=0, mode='constant', cval=0.0)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    inner_magnitudes = gradient_magnitude[inner_mask]

    return np.mean(inner_magnitudes)

def get_outline_boldness(lum, inner_mask, outline_mask):
    """
    The contrast ratios between each outline pixel and its neighboring non-outline pixels
        (unless there are none, in which case the ratios between outline pixels).
    """
    H, W = lum.shape

    adjacents = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    diagonals = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    ratios = []
    weights = []
    for y in range(H):
        for x in range(W):
            if not outline_mask[y, x]:
                continue

            for directions, weight in [(adjacents, 1), (diagonals, 0.707)]:
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and inner_mask[ny, nx]:
                        ratios.append(get_contrast_ratio(lum[y, x], lum[ny, nx]))
                        weights.append(weight)

    if len(ratios) == 0:
        return None, None

    return np.array(ratios), np.array(weights)

def get_pop_factors(lum, opaque_mask):
    touches_transparent = get_touches_transparent_mask(opaque_mask)
    inner_lums = lum[~touches_transparent]
    outline_lums = lum[opaque_mask & touches_transparent]

    if not inner_lums.size == 0:
        pops_inner_light = [get_contrast_ratio(l, LIGHT_BACKGROUND_LUM) for l in inner_lums]
        pops_inner_dark = [get_contrast_ratio(l, DARK_BACKGROUND_LUM) for l in inner_lums]
    else:
        pops_inner_light = [1]
        pops_inner_dark = [1]
        inner_lums = outline_lums
    pops_outline_light = [get_contrast_ratio(l, LIGHT_BACKGROUND_LUM) for l in outline_lums]
    pops_outline_dark = [get_contrast_ratio(l, DARK_BACKGROUND_LUM) for l in outline_lums]

    return inner_lums, outline_lums, pops_inner_light, pops_outline_light, pops_inner_dark, pops_outline_dark

"""
def get_metrics(img):
    img_arr = np.array(img)
    rgb = img_arr[..., :3]
    alpha = img_arr[..., 3]
    opaque_rgb = rgb[alpha == 255]

    brightness = np.mean(opaque_rgb)
    contrast = np.std(opaque_rgb)
    
    color_count = np.unique(opaque_rgb.reshape(-1, 3), axis=0).shape[0]
    edge_density = np.mean(sobel(rgb.mean(axis=2)))
    area = np.mean(alpha == 255)
    transparency = np.mean(alpha < 255)

    hsv = rgb2hsv(rgb / 255)
    hue_hist, _ = np.histogram(hsv[..., 0], bins=12, range=(0, 1))
    
    return {
        "brightness": brightness, # The texture's average brightness (not value), 0 to 255
        "contrast": contrast, # The texture's contrast as a standard deviation of rgb values 
        "color_count": color_count, # The number of distinct opaque color codes in the texture 
        "edge_density": edge_density, # Grayscales, then applies Sobel filter to find the texture's average edge strength
        "area": area, # Scale from 0 to 1 based on the percentage of the texture that is opaque
        "transparency": transparency, # Scale from 0 to 1 based on the percentage of the texture that is transparent
        "hue_histogram": hue_hist.tolist() # Dumps pixels into 12-hue buckets, irrespective of saturation
    }
    # Considerations: 
    # color_clause = More lenient than color count, will blend colors that look similar enough. Only notable for textures with subtle noise
    # color_continuity = Maybe take the deviation of the sizes of 4-contiguous color blocs, proportionally to color_count / color_clause and area

def get_style_profile(directory):
    all_metrics = []
    for fname in os.listdir(directory):
        if fname.endswith(".png"):
            img = load_texture(os.path.join(directory, fname))
            all_metrics.append(get_metrics(img))

    if not all_metrics:
        print("Failed to get style profile")
        return None

    profile = {}
    for measurement in all_metrics[0]:
        if (measurement == "hue_histogram"):
            profile[measurement] = np.mean([m[measurement] for m in all_metrics], axis=0).tolist()
        else:
            profile[measurement] = np.mean([m[measurement] for m in all_metrics])
    
    return profile
"""
