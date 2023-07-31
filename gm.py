import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.set_printoptions(threshold=np.inf)

# latent variables distribution: gaussian mixture
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20, num_classes=10):
        super(VAE, self).__init__()
        self.num_classes = num_classes
        self.z_dim = z_dim

        self.inferwNet = InferwNet(image_size, num_classes)
        self.attentionNet = AttentionNet(image_size + num_classes, h_dim, 800)
        self.image_size = image_size

        # Initialize inferwNet
        for module in self.inferwNet.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        self.inferzNet = torch.nn.ModuleList([
            nn.Linear(image_size + num_classes, h_dim),
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

        self.w_mu = nn.Linear(num_classes, z_dim)
        self.w_var = nn.Linear(num_classes, z_dim)

    def inferw(self, x):
        return self.inferwNet(x, self.image_size)


    def inferz(self, x, w):

        # Compute attention weights
        # attention_weights = self.compute_attention_weights(x, w)  # shape: (batch_size, 1)

        # Weighted concatenation of x and w
        weighted_x = x  # shape: (batch_size, x_dim)
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
        cross_entropy = 3.0 * self.cross_entropy_loss(res['logits'], res['prob'])
        # print(f"cross_entropy:    {cross_entropy}")
        gaussian_loss = self.gaussian_loss(res['sample'], mu, var, prior_mu, prior_var)
        # print(f"gaussian_loss:    {gaussian_loss}")
        # focal_loss = self.focal_loss(res['logits'], res['prob'])
        # print(f"focal_loss:    {focal_loss}")
        kl_part1 = torch.log(prior_var / var)
        kl_part2 = (var + (prior_mu - mu) ** 2) / prior_var - 1
        kl_div = 0.5 * torch.mean(kl_part1 + kl_part2, dim=1)  # sum the KL divergence of each sample
        kl_div = 5.0 * kl_div.mean()
        # print(f"kl_div:        {kl_div}")
        return cross_entropy + kl_div + gaussian_loss

    def log_normal(self, x, mu, var, eps=1e-8):
      var = var + 1e-8
      return -0.5 * torch.sum(
        np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)

    def gaussian_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior):
        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior)
        return loss.mean()

    def cross_entropy_loss(self, logits, targets):
        return -torch.sum(torch.sum(targets * F.log_softmax(logits, dim=-1), dim=-1))

    def focal_loss(self, logits, targets, alpha=0.5, gamma=4):
        ce_loss = self.cross_entropy_loss(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss

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
    def __init__(self, image_size, num_classes):
        super(InferwNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, num_classes)
        )
        self.num_classes = num_classes
        self.batch_norm = nn.BatchNorm1d(num_classes)

    def forward(self, x, image_size):
        # batch_size, len_size = x.size(0), 28
        # reshaped_x = x.view(batch_size, 1, len_size, len_size)
        # # Random rotation
        # rotation_transform = transforms.RandomRotation(degrees=(-50, 50))
        # rotated_x = torch.stack([rotation_transform(img) for img in reshaped_x])

        x_reshape = x.view(-1, 1, 28, 28)
        features = self.conv_layers(x_reshape)
        features = features.view(features.size(0), -1)  # Flatten the tensor
        logits = self.fc_layers(features)
        logits = self.batch_norm(logits).view(-1, self.num_classes)

        batch_size = logits.shape[0]
        max_indices = np.argmax(logits.cpu().detach().numpy(), axis=1)
        prob = torch.zeros_like(logits)
        prob[torch.arange(batch_size), max_indices] = 1

        w = F.gumbel_softmax(logits, tau=1e-3, hard=True)
        # print(f"logits: {logits[0]}")
        # print(f"prob:   {prob[0]}")
        # print(f"gumbel: {w[0]}")
        return logits, prob, w



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