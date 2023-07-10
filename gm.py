import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

# latent variables distribution: gaussian mixture
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20, num_components=10):
        super(VAE, self).__init__()
        self.num_components = num_components
        self.z_dim = z_dim

        self.inferwNet = torch.nn.ModuleList([
            nn.Linear(image_size, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            GumbelSoftmaxLayer(h_dim, num_components)
        ])

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
        for i, layer in enumerate(self.inferwNet):
            x = layer(x)
        return x

    def inferz(self, x, w):
        concat = torch.cat((x, w), dim=1)
        for layer in self.inferzNet:
            concat = layer(concat)
        return concat

    def generateprior(self, w):
        w_mu = self.w_mu(w)
        w_var = F.softplus(self.w_var(w))
        return w_mu, w_var

    def generatex(self, z):
        for layer in self.generatexNet:
          z = layer(z)
        return z

    def encode(self, x):
        logits, w = self.inferw(x)
        mu, var, z = self.inferz(x, w)
        res = {'mean': mu, 'var': var, 'sample': z,
              'logits': logits, 'categorical': w}
        return res

    def decode(self, z, w):
        prior_mu, prior_var = self.generateprior(w)
        x_rec = self.generatex(z)
        return x_rec, prior_mu, prior_var

    def forward(self, x):
        res = self.encode(x)
        x_rec, prior_mu, prior_var = self.decode(res['sample'], res['categorical'])
        res.update({'prior_mean': prior_mu, 'prior_var': prior_var, 'x_rec': x_rec})
        return res

    def kl_divergence(self, res):
        mu = res['mean']
        var = res['var']
        prior_mu = res['prior_mean']
        prior_var = res['prior_var']

        kl_part1 = torch.log(prior_var / var)
        kl_part2 = (var + (prior_mu - mu) ** 2) / prior_var - 1

        kl_div = 0.5 * torch.sum(kl_part1 + kl_part2, dim=1)  # sum the KL divergence of each sample

        return kl_div.mean()


class GumbelSoftmaxLayer(nn.Module):
    def __init__(self, h_dim, w_dim):
        super(GumbelSoftmaxLayer, self).__init__()
        self.logits = nn.Linear(h_dim, w_dim)
        self.w_dim = w_dim
    def gumbelsoftmax(self, logits, temperature=1.0):
        w = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
        return F.softmax(w / temperature, dim=-1)

    def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
        v = torch.rand(shape)
        if is_cuda:
            v = v.cuda()
        return -torch.log(-torch.log(v + eps) + eps)

    def forward(self, x, temperature=1.0):
        logits = self.logits(x).view(-1, self.w_dim)
        w = self.gumbelsoftmax(logits, temperature)
        return logits, w


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