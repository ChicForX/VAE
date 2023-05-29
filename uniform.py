import torch
import torch.nn as nn
import torch.nn.functional as F

# latent variables distribution: gaussian
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)

    # Encoding, learning mean and variance
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))

    def reparameterize(self, mu, log_var):
        eps = torch.rand_like(mu)
        return mu + eps * (torch.exp(0.5 * log_var))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var, z

    @staticmethod
    def kl_divergence(mu, log_var):
        q = mu - torch.exp(0.5 * log_var)
        p = mu + torch.exp(0.5 * log_var)
        kl_div = torch.log(1.0 / (p - q))
        return torch.abs(kl_div.sum())