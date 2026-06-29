"""Image utils."""
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from color_utils import srgb_to_oklab, oklab_dist


class Sprite:
    weighted_adjacencies = [
        (-1,-1,0.707),(-1,0,1),(-1,1,0.707),(0,1,1),
        (1,1,0.707),(1,0,1),(1,-1,0.707),(0,-1,1)
    ]

    """Wrapper for image object depicting a sprite."""
    def __init__(self, img):
        self.srgb_arr = np.asarray(img)
        self.oklab_arr = srgb_to_oklab(self.srgb_arr)
        self.compute_palette()
        self.connectivity_graph = nx.Graph()

    def get_srgb_arr(self):
        """Get image array in color mode srgb."""
        return self.srgb_arr

    def get_oklab_arr(self):
        """Get image array in color mode oklab."""
        return self.oklab_arr

    def display(self):
        """Display image array in color."""
        print_image(self.srgb_arr)

    def display_l(self):
        """
        Display image array in monochrome scaled to luminance in oklab
        (not necessarily luminance-preserving).
        """
        print_oklab_l(self.oklab_arr)

    def compute_palette(self):
        """Compute palette."""
        opaque_mask = self.oklab_arr[:, :, 3] == 255
        opaque_pixels = self.oklab_arr[opaque_mask][:, :3]

        flattened_img = opaque_pixels.reshape(-1, 3)

        colors = [tuple(c) for c in flattened_img]
        self.palette = Counter(colors)

    def get_palette(self):
        """Get palette as Counter object."""
        return self.palette

    def compute_connectivities(self, adj_multiplier=3, contour_multplier=3, similarity_multiplier=2):
        """Compute connectivities."""
        opaque_mask = self.srgb_arr[:, :, 3] == 255
        max_i, max_j, _ = self.oklab_arr.shape

        # spatial adjacency pass
        row_indices, col_indices = np.where(opaque_mask)
        for i, j in zip(row_indices, col_indices):
            u = self.oklab_arr[i, j][:3]

            # check neighbors
            for dir_idx, (di, dj, weight) in enumerate(self.weighted_adjacencies):
                new_i = i + di
                new_j = j + dj
                if new_i<0 or new_j<0 or new_i >= max_i or new_j >= max_j:
                    continue

                v = self.oklab_arr[new_i, new_j]
                if v[3] != 255:
                    continue
                v = v[:3]

                colors_are_same = np.array_equal(u, v)
                if not colors_are_same:
                    self.boost_edge_by_distance(u, v, weight * adj_multiplier)
                    continue

                # colors are same. spatial contour pass
                for idx in [(dir_idx - 1) % 8, dir_idx, (dir_idx + 1) % 8]:
                    di, dj, weight = self.weighted_adjacencies[idx]
                    new_i_2 = new_i + di
                    new_j_2 = new_j + dj
                    if new_i_2<0 or new_j_2<0 or new_i_2 >= max_i or new_j_2 >= max_j:
                        continue

                    w = self.oklab_arr[new_i_2, new_j_2]
                    if w[3] != 255:
                        continue
                    w = w[:3]

                    colors_are_same = np.array_equal(v, w)
                    if not colors_are_same:
                        self.boost_edge_by_distance(v, w, weight * contour_multplier)

        # divide relative adjacencies
        # color similarity pass
        colors = self.palette.keys()
        for u in colors:
            for v in colors:
                if np.array_equal(u, v):
                    continue

                u_tuple, v_tuple = tuple(u), tuple(v)
                if not self.connectivity_graph.has_edge(u_tuple, v_tuple):
                    self.boost_edge_by_distance(u, v, similarity_multiplier)
                    continue

                curr_weight = self.connectivity_graph[u_tuple][v_tuple].get("weight", 0)
                denominator = min(self.palette[u_tuple], self.palette[v_tuple])
                new_weight = curr_weight / (denominator + 0.001)

                self.connectivity_graph.add_edge(u_tuple, v_tuple, weight=new_weight)

        return self.connectivity_graph

    def boost_edge_by_distance(self, u, v, weight):
        """Boost edge by a weighted amount that diminishes with distance in color space."""
        assert len(u) == 3, u
        assert len(v) == 3, v
        assert not np.array_equal(u, v), (u, v)

        u_tuple, v_tuple = tuple(u), tuple(v)
        distance = oklab_dist(u, v, w_L=0.707)

        curr_weight = 0
        if self.connectivity_graph.has_edge(u_tuple, v_tuple):
            curr_weight = self.connectivity_graph[u_tuple][v_tuple].get("weight", 0)
        new_weight = curr_weight + weight / (distance ** 2 + 0.001)

        self.connectivity_graph.add_edge(u_tuple, v_tuple, weight=new_weight)

    def cull_connectivities(self, threshold=3):
        """Remove edges whose weights are below the input threshold."""
        edges_to_remove = [
            (u, v) for u, v, data in self.connectivity_graph.edges(data=True)
            if data.get("weight", 0) < threshold
        ]
        self.connectivity_graph.remove_edges_from(edges_to_remove)



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


def l_flatten(oklab_arr):
    """Return lightness representation from oklab image.""" 
    L = oklab_arr[..., 0]
    alpha = oklab_arr[..., 3]

    return np.stack([L, alpha], axis=-1)


def get_color_positions(img_arr, include_transparent = False):
    """
    Deprecated.
    Return a dict that maps distinct colors to a list of their positions from an image array.
    """
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
