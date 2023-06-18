import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
# latent variables distribution: bernoulli(discrete)
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        # Gumbel-Softmax
        logits = torch.log_softmax(mu, dim=-1)
        gumbel_noise = self.sample_gumbel(logits.size(), device=mu.device)
        gumbel_sample = (logits + gumbel_noise).softmax(dim=-1)
        return gumbel_sample

    def sample_gumbel(self, shape, device):
        uniform = torch.rand(shape).to(device)
        return -torch.log(-torch.log(uniform + 1e-20) + 1e-20)

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var, z

    def kl_divergence(self, mu, log_var):
        # prior: Bernoulli
        prior_distribution = dist.Bernoulli(probs=0.5)
        posterior_distribution = dist.Bernoulli(logits=mu)
        kl_div = dist.kl_divergence(posterior_distribution, prior_distribution).sum()
        return kl_div



