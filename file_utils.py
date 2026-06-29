"""File utils."""
import os
from PIL import Image


def get_image(namespace, fname):
    """
    Return image from path.
    """
    folder = get_item_directory(namespace)
    file_path = os.path.join(folder, fname)
    img = load_texture(file_path)

    if not img:
        print("Image not found.")
        exit(1)

    return img


def get_item_directory(namespace):
    """Return path to /items/ directory of the namespace."""
    return "dataset/" + namespace + "/items/"


def load_texture(path):
    """Loads a texture as-is from the path (textures should natively be RGBA)."""
    return Image.open(path).convert("RGBA")
