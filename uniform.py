import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions as dist
# latent variables distribution: gaussian
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)

        self.prior_min = nn.Parameter(torch.tensor([0.0]))  # Prior minimum value
        self.prior_max = nn.Parameter(torch.tensor([2.0]))  # Prior maximum value
        self.z_dim = z_dim
    # Encoding, learning mean and variance
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))

    def reparameterize(self, mu, log_var):
        eps = torch.rand((1, self.z_dim), device=mu.device)
        return mu + eps * (torch.exp(0.5 * log_var))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_rec = self.decode(z)
        res = {'mean': mu, 'log_var': log_var, 'x_rec': x_rec, 'sample': z}
        return res

    def kl_divergence(self, res):
        mu = res['mean']
        log_var = res['log_var']
        prior_len = self.prior_max - self.prior_min
        q = mu - math.sqrt(3) * torch.exp(0.5 * log_var)
        p = mu + math.sqrt(3) * torch.exp(0.5 * log_var)
        kl_div = torch.log(prior_len / (p - q))
        return torch.abs(kl_div.sum())