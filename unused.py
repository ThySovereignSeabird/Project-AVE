def print_image(img):
    """
    Print a small image in the terminal using ANSI color codes.
    """
    img_arr = np.asarray(img)

    for row in img_arr:
        line = []
        for r, g, b, a in row:
            if a < 128:
                line.append("  ")  # transparent
            else:
                line.append(f"\033[48;2;{r};{g};{b}m  \033[0m")
        print("".join(line))  # newline


def collect_features(namespace):
    """
    #Collect features and assign labels on all items from one namespace; return a list of dicts.
    """
    data = []
    folder = get_item_directory(namespace)

    filenames = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
    for fname in filenames:
        file_path = os.path.join(folder, fname)

        context = {"name": f"{namespace}/{fname}",
                   "group": "train" if namespace == "vanilla" else "test",
                    "outline": outline_label_from_tag(file_path)}
        img = load_texture(file_path)
        data.append(context | get_texture_contrast(img))

    if not data:
        print("No data collected.")
        exit(1)

    return data


# from coloraide import Color

# color = Color("#8f3232").convert("oklab")
# print("Oklab Coordinates:", color.coords())

# Example usage with a single color (e.g., pure sRGB Red)
srgb_color = np.array([0.561, 0.196, 0.196])
oklab_coords = srgb_to_oklab(srgb_color)
print("L (Lightness):", oklab_coords[0])
print("a (Green/Red):", oklab_coords[1])
print("b (Blue/Yellow):", oklab_coords[2])