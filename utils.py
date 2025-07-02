import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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