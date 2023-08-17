import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def graphStructure(nodes, edges):
    G = nx.Graph()

    for node_idx, node_position in enumerate(nodes):
        G.add_node(node_idx, pos=node_position)

    for edge in edges:
        G.add_edge(edge[0], edge[1])

    node_positions = {node: (y, 28-x) for node, (x, y) in nx.get_node_attributes(G, 'pos').items()}  # Swap x and y

    plt.figure(figsize=(8, 8))
    nx.draw(G, pos=node_positions, node_size=50, with_labels=False, edge_color='gray')
    plt.show()
