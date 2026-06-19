"""Texture utils."""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def l_flatten(oklab_arr):
    """Return lightness representation from oklab image.""" 
    L = oklab_arr[..., 0]
    alpha = oklab_arr[..., 3]

    return np.stack([L, alpha], axis=-1)


def print_image(img):
    """
    Print a small image in the terminal using ANSI color codes.
    """
    img_arr = np.asarray(img)

    # Display the array as an image
    plt.figure(figsize=(3, 3))
    plt.imshow(img_arr, cmap="gray")
    plt.axis("off")
    plt.show()


def print_oklab_l(oklab_arr):
    """Return what the computer sees according to how L is scaled.""" 
    L = oklab_arr[..., 0]
    L = np.clip(L * 255.0, 0, 255).astype(np.uint8)
    alpha = oklab_arr[..., 3]
    alpha = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)

    rgb_grayscale = np.stack([L, L, L, alpha], axis=-1)
    print_image(rgb_grayscale)


def get_color_positions(img_arr, include_transparent = False):
    """Return a dict that maps distinct colors to a list of their positions from an image array."""
    color_dict = defaultdict(list)
    height, width, channels = img_arr.shape

    # Iterate through every pixel in the image
    for y in range(height):
        for x in range(width):
            color = tuple(img_arr[y, x])

            if channels == 4 and not include_transparent:
                alpha = color[3]
                if alpha == 0:
                    continue
            color_dict[color].append((y, x))

    return color_dict
