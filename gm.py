import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MixtureSameFamily
# latent variables distribution: gaussian mixture
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20, num_components=2):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)
        self.num_components = num_components

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        # Sample component indices
        indices = torch.randint(low=0, high=self.num_components, size=mu.shape[:-1]).to(mu.device)
        # Construct mixture components
        components = Normal(mu, torch.exp(0.5 * log_var))
        # Sample from the mixture distribution
        z = components.rsample()
        # Select samples based on component indices
        z_selected = torch.gather(z, dim=-1, index=indices.unsqueeze(-1)).squeeze(-1)
        return z_selected

    def decode(self, z):
        print(z.size())
        h = F.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def kl_divergence(self, mu, log_var):
        # Construct mixture components
        components = Normal(mu, torch.exp(0.5 * log_var))
        # Construct mixture distribution
        weights = torch.ones(mu.shape[0], self.num_components).to(mu.device) / self.num_components
        mixture_dist = MixtureSameFamily(weights, components)
        # Compute KL divergence between mixture distribution and prior
        prior_dist = Normal(torch.zeros_like(mu), torch.ones_like(log_var))
        kl_div = torch.distributions.kl_divergence(mixture_dist, prior_dist).sum()
        return kl_div
