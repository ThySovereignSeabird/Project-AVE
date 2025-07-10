from PIL import Image
import numpy as np
import os
from skimage.color import rgb2hsv
from skimage.filters import sobel

# python -m pip install -r requirements.txt

VANILLA_ITEMS_DIRECTORY = "dataset/vanilla/items/"

def main():
    print(get_style_profile("dataset/vanilla/items"))
    # img = load_vanilla("diamond_sword.png")
    # print(get_metrics(img))
    # img.show()
    return 0

def load_texture(path):
    """
    Loads a texture as-is from the path (textures should natively be RGBA).
    """
    img = Image.open(path).convert('RGBA')
    return img

def load_vanilla(fname):
    return load_texture(VANILLA_ITEMS_DIRECTORY + fname)

def get_palette_contrast():
    """
    Get the contrast of a palette.
    """
    pass

def get_metrics(img):
    """
    Input an image to get a dictionary of (admittedly basic) metrics.
    """
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

def get_style_profile(folder):
    all_metrics = []
    for fname in os.listdir(folder):
        if fname.endswith(".png"):
            img = load_texture(os.path.join(folder, fname))
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

    #def avg_metric(key):
    #    return np.mean([m[key] for m in all_metrics])
    #
    #profile = {k: avg_metric(k) for k in all_metrics[0] if k != 'hue_histogram'}
    #profile['hue_histogram'] = np.mean([m["hue_histogram"] for m in all_metrics], axis=0).tolist()
    return profile

if __name__ == "__main__":
    main()

