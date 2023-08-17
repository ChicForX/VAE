import torch
import numpy as np

def image2graph_batch(data_batch, height=28, width=28):

    batch_size, _ = data_batch.shape

    nodes_batch = []
    edges_batch = []

    for batch_idx in range(batch_size):
        data = data_batch[batch_idx].reshape(height, width).cpu().numpy()

        # if pixel is less than 0.6, turns it to -1, otherwise 10
        data = np.where(data < 0.6, -1, 10)

        # padding = 2
        img = np.pad(data, [(2, 2), (2, 2)], "constant", constant_values=(-1))

        cnt = 0
        for i in range(2, height+2):
            for j in range(2, width+2):
                if img[i][j] == 10:
                    img[i][j] = cnt
                    cnt += 1

        edges = []
        nodes = np.zeros((cnt, 2))

        for i in range(2, height+2):
            for j in range(2, width+2):
                if img[i][j] == -1:
                    continue

                filter = img[i - 2:i + 3, j - 2:j + 3].flatten()

                # 8 directions of one node
                filter1 = filter[[6, 7, 8, 11, 13, 16, 17, 18]]

                # position of node
                nodes[filter[12]][0] = i - 2
                nodes[filter[12]][1] = j - 2

                # record edge
                for tmp in filter1:
                    if not tmp == -1:
                        edges.append([filter[12], tmp])

        nodes_batch.append(nodes)
        edges_batch.append(edges)

    return nodes_batch, edges_batch
