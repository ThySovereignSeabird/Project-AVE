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



memo_similarity_matrix = []  # 2d array

# Get data
adjacency_counts = defaultdict(int)
lateral_neighbors = [
            (-1,0),
    (0,-1),         (0,1),
            (1,0)
]
diagonal_neighbors = [
    (-1,-1), (-1,1),
    (1,-1), (1,1)
]
for current_color in unique_colors:
    for (y, x) in color_dict[current_color]:
        for dy, dx in lateral_neighbors:
            ny = y + dy
            nx = x + dx
            maxy = oklab_arr.shape[0]
            maxx = oklab_arr.shape[1]
            if ny<0 or nx<0 or ny >= maxy or nx >= maxx:
                continue

            neighbor = tuple(oklab_arr[ny, nx])
            #if neighbor == current_color:
            #    continue
            if len(neighbor) == 4 and neighbor[3] == 0:
                continue

            adjacency_counts[(current_color, neighbor)] += 1

ramp_matrix = []

adjacent_neighbors = [
    (-1,-1),(-1,0),(-1,1),
    (0,-1),        (0,1),
    (1,-1), (1,0), (1,1)
]



print(lum_diff_matrix[10, 7])
print(distance_matrix[10, 7])
print(lum_diff_matrix[10, 7] / distance_matrix[10, 7])



img = get_image("reimagined", "bowl.png")

%matplotlib inline
print_image(img)

# srgb array
img_arr = np.asarray(img)

# oklab array
oklab_arr = srgb_to_oklab(img_arr)
print_oklab_l(oklab_arr)

%matplotlib widget

oklab_points = oklab_arr.reshape(-1, 4)[:, :-1]
graph_colors(oklab_points)







class UndirectedWeightedGraph:
    def __init__(self):
        self.graph = defaultdict(set)

    def add_node(self, node):
        """Add node."""
        if node not in self.graph:
            self.graph[node] = set()

    def add_edge(self, node1, node2, ma):
        """Add edge."""
        self.add_node(node1)
        self.add_node(node2)
        self.graph[node1].add(node2)
        self.graph[node2].add(node1)

    def __str__(self):
        for node, neighbors in self.graph.items():
            printout += (f"{node}: {list(neighbors)}\n")
        return printout