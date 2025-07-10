from PIL import Image
import os
import csv

from texture_analysis import get_texture_contrast

def load_texture(path):
    """
    Loads a texture as-is from the path (textures should natively be RGBA).
    """
    return Image.open(path).convert('RGBA')

def get_item_directory(namespace):
    return "dataset/" + namespace + "/items/"

def export_all_in_namespace(namespace):
    """
    Collects data on all .png textures in the namespace's folder, then exports data as .csv.
    """
    folder = get_item_directory(namespace)

    # Collect exemplars
    exemplars = set()
    with open(f"{folder}exemplars.txt") as f:
        for line in f:
            exemplars.add(line.strip())

    data = []
    for fname in os.listdir(folder):
        if fname.endswith(".png"):
            img = load_texture(os.path.join(folder, fname))
            collected = collect_data(img, fname)
            collected["exemplar"] = fname in exemplars
            collected["name"] = f"{namespace}/{fname}"
            data.append(collected)

    if not data:
        print("No data collected.")
        exit(1)

    keys = data[0].keys()
    with open(f"{namespace}_item_stats.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

def export_exemplars_in_namespace(namespace):
    """
    Collects data from textures identified by exemplars.txt in the namespace's folder, then exports data as .csv.
    """
    folder = get_item_directory(namespace)
    data = []
    with open(f"{folder}exemplars.txt") as exemplars:
        for line in exemplars:
            fname = line.strip()
            if fname.endswith(".png"):
                img = load_texture(os.path.join(folder, fname))
                data.append(collect_data(img, fname))

    if not data:
        print("No data collected.")
        exit(1)

    keys = data[0].keys()
    with open(f"{namespace}_exemplar_item_stats.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
    

def write_all_in_namespace(namespace):
    """
    Writes the names of all .png textures in the namespace's folder, then exports as .txt.
    """
    folder = get_item_directory(namespace)
    output_file = f"{namespace}_items.txt"

    filenames = sorted(fname for fname in os.listdir(folder) if fname.endswith(".png"))
    with open(output_file, "w") as f:
        for name in filenames:
            f.write(name + "\n")

    print(f"Wrote {len(filenames)} entries to {output_file}")
    
def collect_data(img, fname):
    """
    Collect data from the input texture.
    """
    return get_texture_contrast(img, fname)