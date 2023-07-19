import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
np.set_printoptions(threshold=np.inf)

# latent variables distribution: gaussian mixture
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20, num_components=10):
        super(VAE, self).__init__()
        self.num_components = num_components
        self.z_dim = z_dim

        self.inferwNet = InferwNet(image_size, num_components)
        self.attentionNet = AttentionNet(image_size + num_components, h_dim, 800)
        self.image_size = image_size

        # Initialize inferwNet
        for module in self.inferwNet.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        self.inferzNet = torch.nn.ModuleList([
            nn.Linear(image_size + num_components, h_dim),
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

        self.w_mu = nn.Linear(num_components, z_dim)
        self.w_var = nn.Linear(num_components, z_dim)

    def inferw(self, x):
        return self.inferwNet(x)


    def inferz(self, x, w):
        x = x.view(-1, self.image_size)
        # Compute attention weights
        attention_weights = self.compute_attention_weights(x, w)  # shape: (batch_size, 1)

        # Weighted concatenation of x and w
        weighted_x = attention_weights * x  # shape: (batch_size, x_dim)
        concat = torch.cat((weighted_x, w), dim=1)
        # concat = torch.cat((x, w), dim=1)
        for layer in self.inferzNet:
            concat = layer(concat)
        return concat

    def compute_attention_weights(self, x, w):
        # Compute attention scores
        attention_scores = self.attentionNet(torch.cat((x, w), dim=1))  # shape: (batch_size, 1)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=0)  # shape: (batch_size, 1)

        return attention_weights

    def generateprior(self, w):
        w_mu = self.w_mu(w)
        w_var = F.softplus(self.w_var(w))
        return w_mu, w_var

    def generatex(self, z):
        for layer in self.generatexNet:
          z = layer(z)
        return z

    def encode(self, x):
        logits, prob, w = self.inferw(x)
        mu, var, z = self.inferz(x, w)
        res = {'mean': mu, 'var': var, 'sample': z, 'prob': prob,
              'logits': logits, 'categorical': w}
        return res

    def decode(self, z, w):
        prior_mu, prior_var = self.generateprior(w)
        x_rec = self.generatex(z)
        return x_rec, prior_mu, prior_var

    def forward(self, x):
        res = self.encode(x)
        x_rec, prior_mu, prior_var = self.decode(res['sample'], res['categorical'])
        res.update({'prior_mean': prior_mu, 'prior_var': prior_var, 'x_rec': x_rec, 'x': x})
        return res

    def kl_divergence(self, res):
        mu = res['mean']
        var = res['var']
        prior_mu = res['prior_mean']
        prior_var = res['prior_var']
        kl_part1 = torch.log(prior_var / var)
        kl_part2 = (var + (prior_mu - mu) ** 2) / prior_var - 1
        kl_div = 0.5 * torch.sum(kl_part1 + kl_part2, dim=1)  # sum the KL divergence of each sample
        inferw_loss = 1000 * self.inferwNet.inferw_loss(res['logits'], res['categorical'])
        print(inferw_loss)
        return kl_div.mean() + inferw_loss

    def class_entropy(self, logits, targets):
        log_q = F.log_softmax(logits, dim=-1)
        return -(torch.mean(torch.sum(targets * log_q, dim=-1)) + np.log(0.1))



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

class InferwNet(nn.Module):
    def __init__(self, image_size, num_clusters):
        super(InferwNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, num_clusters)
        )
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, num_clusters))
        self.num_clusters = num_clusters

    def initialize_cluster_centers(self, data_loader, num_clusters, device):
        self.eval()

        # Collect all the samples' fc_output
        fc_outputs = []
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(device)
                fc_output, _, _ = self(inputs)
                fc_outputs.append(fc_output.cpu().detach().numpy())
        fc_outputs = np.concatenate(fc_outputs)

        # Using K-Means clustering algorithm to obtain clustering centers
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(fc_outputs)

        # Copy the value of the cluster center to the model parameters
        self.cluster_centers.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)  # Flatten the tensor
        fc_output = self.fc_layers(features)
        # print(fc_output)

        # Cluster using fc_output
        distances = torch.cdist(fc_output, self.cluster_centers)
        w_labels = torch.argmin(distances, dim=1)
        w = torch.zeros(fc_output.size(0), self.num_clusters).to(fc_output.device)
        w.scatter_(1, w_labels.unsqueeze(1).long(), 1.0)

        # Calculate the probability that each sample belongs to the cluster
        prob = F.softmax(-distances, dim=1)

        # Gumbel Softmax Sampling
        gumbel_w = F.gumbel_softmax(prob, tau=1.0, hard=True)
        # TODO test tau value and return gumbel_w
        return fc_output, prob, w

    def kmeans_loss(self, features, gumbel_w):
        # Compute the distance between each sample's feature and its assigned cluster center
        distances = torch.norm(features.unsqueeze(1) - self.cluster_centers, dim=2)

        # Calculate the k-means loss using the Gumbel-Softmax weights
        loss = 1000 * torch.mean(torch.sum(gumbel_w * distances, dim=1))
        print(f"kmeans_loss {loss}")
        return loss

    def inter_cluster_variance(self):
        loss = torch.var(self.cluster_centers, dim=0).sum()
        print(f"inter_loss {loss}")
        return loss

    def intra_cluster_variance(self, features, w):
        # Compute cluster labels using w
        cluster_labels = torch.argmax(w, dim=1)
        cluster_variances = []
        for i in range(self.cluster_centers.size(0)):
            cluster_points = features[cluster_labels == i]
            if cluster_points.size(0) > 1:
                cluster_variance = torch.var(cluster_points, dim=0)
                if torch.isnan(cluster_variance).any():
                    print(f"Cluster {i}: Found nan in cluster_variance")
                    print(f"Cluster {i}: cluster_points: {cluster_points}")
                cluster_variances.append(cluster_variance)

        # Filter out empty clusters before computing mean
        cluster_variances = [variance for variance in cluster_variances if variance.numel() > 0]

        loss = 10000 * torch.stack(cluster_variances).mean()
        print(f"intra_loss {loss}")
        return loss

    def inferw_loss(self, features, gumbel_w):
        return self.kmeans_loss(features, gumbel_w) - \
            self.inter_cluster_variance() + self.intra_cluster_variance(features, gumbel_w)

class AttentionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, scale_factor):
        super(AttentionNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.scale_factor * x
        return x