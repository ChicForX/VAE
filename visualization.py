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

def plot_loss_curve(losses, num_epochs, train_loader, percentiles=[10, 30, 50, 70, 90]):
    plt.plot(losses, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('VAE Convergence')
    plt.legend()

    total_iterations = num_epochs * len(train_loader)
    percentile_iters = [int(p * total_iterations / 100) for p in percentiles]
    percentile_losses = [losses[i] for i in percentile_iters]

    for p, loss in zip(percentiles, percentile_losses):
        plt.annotate(f'{p}%: {loss:.4f}', xy=(percentile_iters[percentiles.index(p)], loss),
                     xytext=(percentile_iters[percentiles.index(p)] + 500, loss + 0.02),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    plt.show()
