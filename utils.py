#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 07:59:57 2025

@author: sako
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from itertools import combinations
from sklearn.manifold import MDS
import igraph as ig

def plot_spring_layout(adj_matrix,
               node_size = 1000,
               font_size = 10,       
               node_labels=[],
               node_colors=[]):
    """
    Parameters:
    ------------
    adj_matrix : 2D array-like
        The adjacency matrix representing the graph. Should be square (n x n) and made of 0-s & 1-s 
    
    node_size : int, optional (default=1000)
        Size of the nodes in the graph visualization.
    
    font_size : int, optional (default=10)
        Size of the font used for node labels.
    
    node_labels : list of str, optional (default=[])
        Labels for each node. If empty, nodes will be labeled with their index.
    
    node_colors : list of str or color values, optional (default=[])
        Colors for each node. If empty, default coloring is used.
    
    Usage:
    -------
    plot_spring_layout(adj_matrix,
                       node_size=800,
                       font_size=12,
                       node_labels=["A", "B", "C"],
                       node_colors=["red", "green", "blue"])
    """
    
    # Create graph from adjacency matrix
    G = nx.from_numpy_array(adj_matrix)
    
    # Relabel nodes using your predefined labels
    if len(node_labels)>0: 
        mapping = {i: label for i, label in enumerate(node_labels)}
        G = nx.relabel_nodes(G, mapping)
        label_flag = True
    else:
        label_flag = False
    
    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=label_flag,node_size=1000, node_color= node_colors,font_size=font_size )
    plt.show()

def plot_multi_graph(graph_list,graph_names = [] ,graph_colors=[]):

    # Apply color and label attributes
    for g, color, name in zip(graph_list, graph_colors, graph_names):
        g.vs["color"] = color
        g.vs["label"] = [str(v.index) for v in g.vs]  # label by index
        g["name"] = name  # optional metadata
    
    # Plotting all graphs side-by-side using Matplotlib
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, (g, ax, title) in enumerate(zip(graph_list, axs, graph_names)):
        ig.plot(
            g,
            target=ax,
            vertex_color=g.vs["color"],
            vertex_label=g.vs["label"],
            layout=g.layout("circle"),
            margin=30
        )
        ax.set_title(title)
    
    # Add custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=name,
                              markerfacecolor=color, markersize=10)
                       for name, color in zip(graph_names, graph_colors)]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=3)
    plt.tight_layout()
    plt.show()


def plot_complexes(points, eps=0.5, draw_rips=True, draw_cech=True, limits_native=False, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    D = distance_matrix(points, points)
    ax.set_aspect('equal')

    if limits_native:
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        x_mean, y_mean = points.mean(axis=0)
        ax.set_xlim(xmin - x_mean, xmax + x_mean)
        ax.set_ylim(ymin - y_mean, ymax + y_mean)

    ax.set_title(f"ε = {eps:.2f}")

    # Plot points
    ax.plot(points[:, 0], points[:, 1], 'o', color='black', markersize=8)

    # Disks
    for p in points:
        circ = Circle(p, eps, fill=False, linestyle='dashed', color='gray', alpha=0.3)
        ax.add_patch(circ)

    
    # Čech
    if draw_cech:

        for i, j in combinations(range(len(points)), 2):
            if D[i, j] <= eps:
                ax.plot(*zip(points[i], points[j]), color='blue', lw=2, alpha=0.6)
        for i, j, k in combinations(range(len(points)), 3):
            if D[i, j] <= eps and D[i, k] <= eps and D[j, k] <= eps:
                triangle = np.array([points[i], points[j], points[k]])
                ax.fill(triangle[:, 0], triangle[:, 1], color='blue', alpha=0.15)
                
    # Vietoris–Rips
    if draw_rips:
        for i, j in combinations(range(len(points)), 2):
            if D[i, j] <= 2 * eps:
                ax.plot(*zip(points[i], points[j]), color='green', lw=2, alpha=0.6)
        for i, j, k in combinations(range(len(points)), 3):
            pi, pj, pk = points[i], points[j], points[k]
            a, b, c = D[i, j], D[i, k], D[j, k]
            s = (a + b + c) / 2
            try:
                area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                radius = (a * b * c) / (4 * area)
                if radius <= eps:
                    triangle = np.array([pi, pj, pk])
                    ax.fill(triangle[:, 0], triangle[:, 1], color='green', alpha=0.15)
            except:
                continue

    ax.grid(True)
 

def fix_origin(coords, fixed_index=0):
    """
    Translate the coordinates so that the fixed_index point is at (0, 0)
    """
    return coords - coords[fixed_index]

def rotate_to_align(coords, align_index=1):
    """
    Rotate the configuration so that the align_index point lies on the x-axis
    """
    vec = coords[align_index]
    angle = -np.arctan2(vec[1], vec[0])
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle),  np.cos(angle)]])
    return coords @ rot_matrix.T

def embed_points_from_distances(distance_matrix):
    """
    Embed points in 2D space using classical MDS
    """
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(distance_matrix)

    # Post-process: fix first point at origin and align second on x-axis
    coords = fix_origin(coords, fixed_index=0)
    coords = rotate_to_align(coords, align_index=1)
    return coords