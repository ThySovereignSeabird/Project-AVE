import numpy as np
from sys import argv
import pprint

# python -m pip install -r requirements.txt

from texture_analysis import get_texture_contrast
from file_handler import load_texture, get_item_directory, export_namespace, export_all, rollcall_namespace

def main():
    if len(argv) == 1: # python ave.py 
        test_vanilla_item("amethyst_shard.png")
        test_vanilla_item("diamond_sword.png")
        test_vanilla_item("coal.png")
        test_vanilla_item("kelp.png")
        test_vanilla_item("minecart.png")
        test_item("asve", "minecart.png")
        test_vanilla_item("recovery_compass_00.png")
        test_texture("ignore/test_0.png")

    elif len(argv) == 2: 
        if argv[1] == "export": # python ave.py export. Exports data for every namespace with a .csv in labels
            export_all()

        else: # python ave.py [texturepath.png] 
            test_texture(argv[1])

    elif len(argv) == 3:
        if argv[1] == "compare": # python ave.py compare [item.png]. Compares vanilla with ASVE
            test_vanilla_item(argv[2])
            test_item("asve", argv[2])

        elif argv[2] == "rollcall": # python ave.py [namespace] rollcall. Writes .txt of all items in namespace
            rollcall_namespace(argv[1])

        elif argv[2] == "export": # python ave.py [namespace] export. Exports a namespace with a .txt in labels
            export_namespace(argv[1])

        else: # python ave.py [namespace] [item.png]
            test_item(argv[1], argv[2])

def test_vanilla_item(fname):
    test_item("vanilla", fname)

def test_item(namespace, fname):
    img = load_texture(get_item_directory(namespace) + fname)
    perform_tests(img, f"{namespace}/{fname}")
    
def test_texture(fname):
    img = load_texture(fname)
    perform_tests(img, fname)

def perform_tests(img, print_title):
    """
    Perform and print tests.
    """
    print_texture(img)
    print(print_title)
    pprint.pprint(get_texture_contrast(img))
    # img.show()

def print_texture(img):
    """
    Print a small image in the terminal using ANSI color codes.
    """
    img_arr = np.array(img)
    for row in img_arr:
        for pixel in row:
            r, g, b, a = pixel
            if a < 128:
                print("  ", end='') # transparent
            else:
                print(f"\033[48;2;{r};{g};{b}m  \033[0m", end='') # opaque color block
        print() # newline

if __name__ == "__main__":
    main()