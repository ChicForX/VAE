import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions as dist

# latent variables distribution: uniform
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20, half_len=1.0):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)
        self.prior_min = -half_len
        self.prior_max = half_len

    # Encoding, learning mean and log_var
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        eps = torch.rand_like(mu)
        a = mu - math.sqrt(3) * torch.exp(log_var / 2)
        b = mu + math.sqrt(3) * torch.exp(log_var / 2)
        return a + (b - a) * eps

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_rec = self.decode(z)
        res = {'mean': mu, 'log_var': log_var, 'x_rec': x_rec, 'sample': z}
        return res

    def kl_divergence(self, res):
        log_var = res['log_var']
        prior_len = self.prior_max - self.prior_min
        posterior_len = 2 * math.sqrt(3) * torch.exp(log_var / 2)
        eps = 1e-8  # Small epsilon for numerical stability
        kl_div = torch.log(prior_len / (posterior_len + eps))
        return torch.abs(kl_div.sum())

