import os

import matplotlib.pyplot as plt
import squarify


def directory_size(directory):
    total_size = 0
    for path, dirs, files in os.walk(directory):
        for f in files:
            fp = os.path.join(path, f)
            total_size += os.path.getsize(fp)
    return total_size


def analyze_directory(directory):
    size_dict = dict()
    for item in os.listdir(directory):
        # check if item is a directory
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            size = directory_size(item_path)
            if size > 1e6:
                print(f"directory {item}: {size/1e6} MB")
                size_dict[item + f", {size/1e6:.2f} MB"] = size
        # check if item is a file
        elif os.path.isfile(item_path):
            size = os.path.getsize(item_path)
            if size > 1e6:
                print(f"file {item}: {size/1e6} MB")
                size_dict[item + f", {size/1e6:.2f} MB"] = size
    return size_dict


colors = ["darkcyan", "lightgreen", "lightsteelblue", "lightskyblue"]


def plot_tree_map(size_dict):
    sizes = size_dict.values()
    labels = size_dict.keys()
    squarify.plot(sizes=sizes, label=labels, alpha=0.7)
    plt.savefig("Desktop_tree_map.pdf")


path_to_desktop = os.path.join(os.path.expanduser("~"), "Desktop")
size_dict = analyze_directory(path_to_desktop)
plot_tree_map(size_dict)
