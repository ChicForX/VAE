import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.set_printoptions(threshold=np.inf)
import preconditioning_graph
from torch_geometric.nn import GCNConv
import visualization

# latent variables distribution: graph structured
# map x to graph, then input graph to encoder(GCN)
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=70, node_sampled=70, gcn_output=48):
        super(VAE, self).__init__()
        self.node_sampled = node_sampled
        self.z_dim = z_dim
        self.infergNet = GraphNet(gcn_output)
        self.image_size = image_size

        self.inferzNet = torch.nn.ModuleList([
            nn.Linear(node_sampled * gcn_output, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            ReparameterizeTrick(h_dim, z_dim)
        ])

        self.generatexNet = torch.nn.ModuleList([
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, image_size),
            torch.nn.Sigmoid()
        ])

        self.w_mu = nn.Linear(node_sampled * gcn_output, z_dim)
        self.w_var = nn.Linear(node_sampled * gcn_output, z_dim)

    def inferg(self, x):
        return self.infergNet(x)


    def inferz(self, g):
        concat = g
        for layer in self.inferzNet:
            concat = layer(concat)
        return concat


    def generatex(self, z):
        for layer in self.generatexNet:
          z = layer(z)
        return z

    def encode(self, x):
        g = self.inferg(x)
        mu, var, z = self.inferz(g)
        res = {'mean': mu, 'var': var, 'sample': z, 'graph': g}
        return res

    def decode(self, z):
        x_rec = self.generatex(z)
        return x_rec

    def forward(self, x):
        res = self.encode(x)
        x_rec = self.decode(res['sample'])
        res.update({'x_rec': x_rec, 'x': x})
        return res

    def kl_divergence(self, res):
        mu = res['mean']
        var = res['var']
        kl_divergence = - 0.5 * torch.mean(1 + torch.log(var) - mu.pow(2) - var)
        return kl_divergence


class ReparameterizeTrick(nn.Module):

    def __init__(self, h_dim, z_dim):
        super(ReparameterizeTrick, self).__init__()
        self.mu = nn.Linear(h_dim, z_dim)
        self.var = nn.Linear(h_dim, z_dim)

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu = self.mu(x)
        var = F.softplus(self.var(x))
        z = self.reparameterize(mu, var)
        return mu, var, z


class GraphNet(torch.nn.Module):
    def __init__(self, gcn_output):
        super(GraphNet, self).__init__()
        self.conv1 = GCNConv(2, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, gcn_output)


    def forward(self, data):
        batch_size, _ = data.size()
        nodes_batch, edges_batch = preconditioning_graph.image2graph_batch(data)
        # visualization.graphStructure(nodes_batch[0], edges_batch[0])
        x_list = []

        for i in range(batch_size):
            nodes = torch.tensor(nodes_batch[i], dtype=torch.float, device=data.device)
            edges = torch.tensor(edges_batch[i], dtype=torch.long, device=data.device).t()

            edge_index = torch.stack([edges[0], edges[1]], dim=0)
            x = self.conv1(nodes, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.conv3(x, edge_index)
            x = F.relu(x)

            # sample 70 nodes
            num_nodes = x.size(0)
            logits = torch.randn(num_nodes, 70, device=data.device)
            sampled_indices = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(0.5,
                                                                                               logits=logits).rsample()
            x_sampled = torch.matmul(sampled_indices.transpose(0, 1), x)

            if x_sampled.size(0) < 70:
                padding_size = 70 - x_sampled.size(0)
                x_padding = torch.zeros(padding_size, x_sampled.size(1), device=data.device)
                x_sampled = torch.cat([x_sampled, x_padding], dim=0)

            x_list.append(x_sampled)

        x_batch = torch.stack(x_list, dim=0)  # (batch_size, 70, 48)
        x_batch = x_batch.view(batch_size, -1)  # reshape to (batch_size, 70 * 48)
        return x_batch
