#!/usr/bin/env python
# coding: utf-8
import os

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from wildlife_tools.data import FeatureDataset


def _group_type_from_name(group_name):
    """Extract group type prefix from a group name like 'MCP01' -> 'MCP'."""
    return "".join([c for c in group_name if not c.isdigit()])


def _compute_node_sizes(valid_dphids, dphid_counts, min_size=300, max_size=2000):
    """Compute node sizes proportional to dolphin occurrence counts.

    Node size in networkx represents area, so square-root scaling is used.

    Parameters
    ----------
    valid_dphids : array-like
        List of DphID values.
    dphid_counts : pandas.Series or dict
        Mapping from DphID to occurrence count.
    min_size, max_size : float
        Minimum and maximum node sizes.

    Returns
    -------
    list[float]
        Node sizes in the same order as valid_dphids.
    """
    counts = np.array([dphid_counts.get(d, 1) for d in valid_dphids], dtype=float)
    min_count, max_count = counts.min(), counts.max()
    if max_count > min_count:
        normalized = (np.sqrt(counts) - np.sqrt(min_count)) / (
            np.sqrt(max_count) - np.sqrt(min_count)
        )
    else:
        normalized = np.zeros_like(counts)
    return (min_size + normalized * (max_size - min_size)).tolist()


def plot_social_network(root_dir, relationship_matrices, valid_dphids, node_sizes=None):
    """Plot dolphin social relationship network graph with edges colored by group type.

    Parameters
    ----------
    root_dir : str or Path
        Root directory for saving the output image.
    relationship_matrices : dict[str, numpy.ndarray]
        Mapping from group type (e.g. 'MCP', 'NN', 'SYN') to its adjacency matrix.
    valid_dphids : array-like
        List of DphID values corresponding to matrix rows/columns.
    node_sizes : array-like, optional
        Size of each node. If None, all nodes use size 500.
    """
    root_dir = str(root_dir)
    n_dphids = len(valid_dphids)

    if node_sizes is None:
        node_sizes = [500] * n_dphids
    else:
        node_sizes = list(node_sizes)
        if len(node_sizes) != n_dphids:
            raise ValueError("node_sizes length must match valid_dphids length")

    # Combined matrix for graph layout
    combined_matrix = sum(relationship_matrices.values())

    G = nx.Graph()
    for dphid in valid_dphids:
        G.add_node(dphid)

    for i in range(n_dphids):
        for j in range(i + 1, n_dphids):
            weight = combined_matrix[i, j]
            if weight > 0:
                G.add_edge(valid_dphids[i], valid_dphids[j], weight=weight)

    plt.figure(figsize=(16, 16))
    if len(G.edges) > 0:
        # Increase repulsion (k) so nodes do not overlap
        k = max(2.0, 3.0 / np.sqrt(n_dphids))
        pos = nx.spring_layout(G, weight="weight", seed=42, k=k, iterations=50)
    else:
        pos = nx.circular_layout(G)

    color_map = {
        "MCP": "#e41a1c",  # red
        "NN": "#377eb8",   # blue
        "SYN": "#4daf4a",  # green
    }
    default_colors = ["#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf"]

    group_types = sorted(relationship_matrices.keys())
    legend_handles = []

    edge_label_infos = []

    for idx, group_type in enumerate(group_types):
        matrix = relationship_matrices[group_type]
        color = color_map.get(group_type, default_colors[idx % len(default_colors)])

        edges = []
        edge_weights = []
        for i in range(n_dphids):
            for j in range(i + 1, n_dphids):
                weight = matrix[i, j]
                if weight > 0:
                    edges.append((valid_dphids[i], valid_dphids[j]))
                    edge_weights.append(weight)

        if not edges:
            continue

        max_weight = max(edge_weights) if max(edge_weights) > 0 else 1
        widths = [2 + 4 * w / max_weight for w in edge_weights]

        # Stagger curved edges so overlapping edge types remain visible
        if len(group_types) > 1:
            curvature = 0.15 * (idx - (len(group_types) - 1) / 2)
            connectionstyle = f"arc3,rad={curvature}"
        else:
            curvature = 0.0
            connectionstyle = "arc3,rad=0"

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            width=widths,
            alpha=0.7,
            edge_color=color,
            connectionstyle=connectionstyle,
            arrows=True,
            arrowstyle="-",
            node_size=node_sizes,
        )

        edge_label_infos.append((edges, edge_weights, curvature, color))

        legend_handles.append(
            Line2D([0], [0], color=color, lw=2, label=group_type)
        )

    if legend_handles:
        plt.legend(handles=legend_handles, loc="upper right", title="Group Type")

    # Draw nodes on top of edges so they are not overlaid
    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes, node_color="skyblue", edgecolors="white"
    )

    # Draw node labels
    nx.draw_networkx_labels(
        G, pos, labels={d: "DphID%03d" % d for d in valid_dphids}, font_size=16
    )

    # Draw edge weight labels in the middle of edges
    # Reference: https://networkx.org/documentation/stable/auto_examples/drawing/plot_weighted_graph.html
    for edges, edge_weights, curvature, color in edge_label_infos:
        connectionstyle = f"arc3,rad={curvature}" if curvature != 0 else "arc3,rad=0"
        edge_labels = {(u, v): "%d"%int(w) for (u, v), w in zip(edges, edge_weights)}
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=14,
            font_color=color,
            connectionstyle=connectionstyle,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=1,
            ),
        )

    plt.title("Dolphin Social Network")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(root_dir, "METAINFO", "FIN_SOCIAL_NETWORK.png")
    )
    plt.close()


def statistics(root_dir):
    """Generate fin matching statistics summary and social network graph.

    Parameters
    ----------
    root_dir : str or Path
        Root directory of the dataset.

    Returns
    -------
    tuple (FeatureDataset, dict[str, numpy.ndarray])
        The loaded feature dataset and the per-group-type social adjacency matrices.
    """
    root_dir = str(root_dir)
    features = FeatureDataset.from_file(
        os.path.join(
            root_dir, "METAINFO", "FIN_DEEPPFEATUES_SELECTED_MERGED_PAIRED_SOCIAL"
        )
    )
    METAINFO_csv = os.path.join(root_dir, "METAINFO", "FIN_METAINFO.csv")
    full_metainfo = pd.read_csv(METAINFO_csv, index_col=0)

    # Plot Statistics
    FinID_hist = features.metadata.FinID2.value_counts()
    DphID_hist = features.metadata.DphID.value_counts()
    DphID_count = len(features.metadata.DphID.unique()) - 1
    relationship_hist = features.metadata.confirmed_group.value_counts()
    captureability = features.metadata.DphID.value_counts()
    DphID_list = ["DphID %d" % id for id in captureability.index.values]

    orignal_image_number = len(full_metainfo.orig_img.unique())
    has_dphid_image_number = len(
        features.metadata.orig_img[features.metadata.DphID != 0].unique()
    )

    fig, axd = plt.subplot_mosaic(
        [["FinID", "DphID", "Captureablity"],
         ["Social", "Social", "Social"]],
        figsize=(16, 9),
    )
    fig.suptitle(
        "GROUP MATCHING STATISTICS SUMMARY\n\n"
        + "%s\n" % (root_dir.split("/")[-2])
        + "Assign %d images with %d DphID in total %d images"
        % (has_dphid_image_number, DphID_count, orignal_image_number)
    )

    # Plot Fin ID Histogram
    axd["FinID"].bar(range(len(FinID_hist)), FinID_hist.values)
    axd["FinID"].set_xticks(
        range(len(FinID_hist)),
        labels=["%02d" % i for i in FinID_hist.index.values],
        rotation=90,
        fontsize=5.5,
    )
    axd["FinID"].set_xlabel("FinID Assigned to DphID")
    axd["FinID"].set_ylabel("Count")
    # Plot Dph ID Histogram
    axd["DphID"].bar(range(len(DphID_hist)), DphID_hist.values)
    axd["DphID"].set_xticks(
        range(len(DphID_hist)),
        ["%02d" % i for i in DphID_hist.index.values],
        rotation=90,
        fontsize=7,
    )
    axd["DphID"].set_xlabel("DphID")
    axd["DphID"].set_ylabel("Count")

    # Plot Captureablity
    axd["Captureablity"].pie(
        captureability.values, labels=DphID_list, autopct="%1.1f%%"
    )

    # Plot Soical Relationship
    axd["Social"].bar(
        relationship_hist.index.values[1:], relationship_hist.values[1:]
    )
    axd["Social"].set_xticks(
        relationship_hist.index.values[1:],
        rotation=90,
        labels=relationship_hist.index.values[1:],
    )
    axd["Social"].set_xlabel("SOCIAL RELATIONSHIP")
    axd["Social"].set_ylabel("Occurred Number")

    plt.savefig(
        os.path.join(root_dir, "METAINFO", "FIN_STASTISTICS_SUMMARY.png")
    )

    # Build social adjacency matrices per group type
    valid_dphids = sorted(
        features.metadata.loc[features.metadata.DphID != 0, "DphID"].unique()
    )
    dphid_to_idx = {dphid: idx for idx, dphid in enumerate(valid_dphids)}
    n_dphids = len(valid_dphids)

    relationship_matrices = {}

    grouping_info = features.metadata.loc[features.metadata.confirmed_group != ""]
    group_list = grouping_info.confirmed_group.unique()

    for group_name in group_list:
        group_type = _group_type_from_name(group_name)
        if group_type not in relationship_matrices:
            relationship_matrices[group_type] = np.zeros((n_dphids, n_dphids))

        group_rows = grouping_info.loc[grouping_info.confirmed_group == group_name]
        fin_count = len(group_rows)
        group_dphids = group_rows["DphID"].unique()
        if len(group_dphids) == 0:
            continue
        occurred_number = fin_count / len(group_dphids)
        for i in group_dphids:
            for j in group_dphids:
                if i in dphid_to_idx and j in dphid_to_idx:
                    idx_i, idx_j = dphid_to_idx[i], dphid_to_idx[j]
                    relationship_matrices[group_type][idx_i, idx_j] += occurred_number

    # Save per-group-type adjacency matrices as CSV
    for group_type, matrix in relationship_matrices.items():
        relationship_df = pd.DataFrame(
            matrix,
            index=["DphID%03d" % d for d in valid_dphids],
            columns=["DphID%03d" % d for d in valid_dphids],
        )
        relationship_df.to_csv(
            os.path.join(
                root_dir, "METAINFO", f"FIN_SOCIAL_ADJACENCY_MATRIX_{group_type}.csv"
            )
        )

    # Save combined adjacency matrix as CSV
    combined_matrix = sum(relationship_matrices.values())
    combined_df = pd.DataFrame(
        combined_matrix,
        index=["DphID%03d" % d for d in valid_dphids],
        columns=["DphID%03d" % d for d in valid_dphids],
    )
    combined_df.to_csv(
        os.path.join(root_dir, "METAINFO", "FIN_SOCIAL_ADJACENCY_MATRIX.csv")
    )

    # Compute node sizes based on dolphin occurrence count
    #TODO: only count dolphin in different view, rather than repeat count in boot shooting
    dphid_counts = features.metadata.loc[
        features.metadata.DphID != 0, "DphID"
    ].value_counts()
    node_sizes = _compute_node_sizes(valid_dphids, dphid_counts)

    # Plot the social network graph
    plot_social_network(
        root_dir, relationship_matrices, valid_dphids, node_sizes=node_sizes
    )

    return features, relationship_matrices


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate fin matching statistics summary and social network graph."
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        default="/media/filming/2025-白海豚/20240825-JM_02-3/",
        help="Root directory of the dataset",
    )
    args = parser.parse_args()
    statistics(args.root_dir)
