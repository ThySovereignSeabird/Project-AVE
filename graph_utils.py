"""Graph utils."""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from color_utils import oklab_to_srgb


def print_color_graph(graph):
    """Print color graph nodes and edges."""
    fig, ax = plt.subplots(figsize=(6, 6))
    pos = nx.kamada_kawai_layout(graph)

    # Create a node colors list from color groups
    node_colors = []
    for node in graph.nodes:
        ok_l, ok_a, ok_b = node
        r, g, b = oklab_to_srgb(np.asarray([ok_l, ok_a, ok_b]))
        node_colors.append(f"#{r:02X}{g:02X}{b:02X}")

    # Draw all nodes at once with their respective colors
    nx.draw_networkx_nodes(
        graph, pos,
        node_size=500,
        node_color=node_colors,
        ax=ax
    )

    # Separate relevant and irrelevant edges
    relevant_edges = [(u, v) for (u, v) in graph.edges if graph.edges[u, v]["weight"] >= 40]
    irrelevant_edges = [(u, v) for (u, v) in graph.edges if not graph.edges[u, v]["weight"] >= 40]

    # Draw relevant edges in green
    if relevant_edges:
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=relevant_edges,
            edge_color='green',
            width=1.5,
            alpha=0.6,
            ax=ax
        )

    # Draw irrelevant edges in red
    if irrelevant_edges:
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=irrelevant_edges,
            edge_color='red',
            width=1.5,
            alpha=0.6,
            ax=ax
        )

    ax.set_axis_off()


def extract_ramps(graph, max_length=25):
    """Extract ramps."""
    ramps = []

    # Sort by L to optimize
    sorted_nodes = sorted(
        graph.nodes,
        key=lambda color: color[0],
        reverse=True
    )

    def depth_first_construction(node, prefix):
        dead_end = True
        for edge in graph.edges(node):
            neighbor = edge[1]
            if neighbor in prefix:
                continue
            # condition will be set
            if not condition:
                continue

            dead_end = False
            depth_first_construction(neighbor, prefix + [neighbor])

        if dead_end:
            ramps.append(prefix)

    # depth-first search
    for node in sorted_nodes:
        depth_first_construction(node, [node])

    return ramps


def is_monotonic_direction(deltas):
    # remove near-zero steps for lenience & to prevent false negatives from noise
    non_zero = deltas[np.abs(deltas) > 0.05]
    if len(non_zero) == 0:
        return True
    return np.all(non_zero <= 0) or np.all(non_zero >= 0)


def is_monotonic_delta_e(delta_e_steps, tolerance=1):
    diffs = np.diff(delta_e_steps)
    non_zero_diffs = diffs[np.abs(diffs) > tolerance]
    if len(non_zero_diffs) == 0:
        return True  # Flat ΔE changes, accept as monotonic
    return np.all(non_zero_diffs >= 0) or np.all(non_zero_diffs <= 0)


def is_consistent_step_size(deltas, tolerance):
    non_zero = deltas[np.abs(deltas) > 0.001]
    if len(non_zero) < 2:
        return True
    step_diffs = np.diff(non_zero)
    return np.all(np.abs(step_diffs) <= tolerance)


def is_subsequence_of_any(candidate, ramps):
    candidate_length = len(candidate)
    reversed_candidate = list(reversed(candidate))

    for ramp in ramps:
        ramp_length = len(ramp)
        if ramp_length <= candidate_length:
            continue
        for i in range(ramp_length - candidate_length + 1):
            if ramp[i:i + candidate_length] == candidate or ramp[i:i + candidate_length] == reversed_candidate:
            ## if ramp[i:i + candidate_length] == candidate:
                return True
    return False