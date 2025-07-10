from PIL import Image
import os
import csv

from texture_analysis import get_texture_contrast_old, get_texture_contrast

LABELS_DIRECTORY = "labels/"

def load_texture(path):
    """
    Loads a texture as-is from the path (textures should natively be RGBA).
    """
    return Image.open(path).convert('RGBA')

def get_item_directory(namespace):
    return "dataset/" + namespace + "/items/"

def find_dict_by_name(list_of_dicts, name):
    return next((d for d in list_of_dicts if d.get("name") == name), None)

def export_namespace(namespace):
    """
    Collects data on all .png textures in the namespace's folder, then exports data as .csv.
    """
    # Collect label data
    label_data = []
    labels_path = f"labels/{namespace}.csv"
    try:
        with open(labels_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                label_data.append(row)
    except FileNotFoundError:
        print(f"Error: The file '{labels_path}' was not found.")
        exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

    export_data(collect_data(label_data, namespace), namespace)

def export_all():
    """
    Collects data on all namespaces with .csvs in labels/, then exports collective data as .csv.
    """
    data = []
    # Collect label data
    try:
        label_files = sorted(f for f in os.listdir(LABELS_DIRECTORY) if f.endswith(".csv"))
        for label_file in label_files:
            label_data = []
            with open(LABELS_DIRECTORY + label_file, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    label_data.append(row)
            namespace = os.path.splitext(label_file)[0]
            data += collect_data(label_data, namespace)
    except FileNotFoundError:
        print(f"Error: No .csv in labels/ was found.")
        exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

    export_data(data, "all")

def collect_data(label_data, namespace):
    """
    Collect the data on all items from one namespace.
    """
    data = []
    folder = get_item_directory(namespace)

    filenames = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
    for fname in filenames:
        label_dict = find_dict_by_name(label_data, fname)
        context = {"name": f"{namespace}/{fname}",
                    "rating": label_dict.get("rating", "unsightly"),
                    "multicolor": label_dict.get("multicolor", False)}
        img = load_texture(os.path.join(folder, fname))
        data.append(context | get_texture_contrast(img))

    if not data:
        print("No data collected.")
        exit(1)

    return data

def export_data(data, title):
    keys = data[0].keys()
    with open(f"output/{title}_item_stats.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
  
def rollcall_namespace(namespace):
    """
    Writes the names of all .png textures in the namespace's folder, then exports as .csv.
    """
    folder = get_item_directory(namespace)
    output_file = f"output/{namespace}_rollcall.csv"

    filenames = sorted(fname for fname in os.listdir(folder) if fname.endswith(".png"))
    with open(output_file, "w") as f:
        for name in filenames:
            f.write(name + "\n")

    print(f"Wrote {len(filenames)} entries to {output_file}")