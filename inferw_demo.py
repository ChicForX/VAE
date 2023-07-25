import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
np.set_printoptions(threshold=np.inf)
batch_size = 128
# Get the dataset with normalization and standardization
transform = transforms.Compose([
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize the data with mean and standard deviation of MNIST
])

dataset = torchvision.datasets.MNIST(root='./data',
                                     train=True,
                                     transform=transform,  # Apply the defined transform
                                     download=True)

# Load data by batch size and shuffle randomly
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

def main():

    num_clusters = 10
    inferw_net = InferwNet(image_size=28*28, num_clusters=num_clusters)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inferw_net.to(device)

    # init cluster centers
    inferw_net.initialize_cluster_centers(data_loader, num_clusters, device)

    # Visualize with t-SNE
    # visualize_with_tsne(inferw_net, data_loader, device)

    num_epochs = 5
    learning_rate = 1e-8
    train_inferw_net(inferw_net, data_loader, num_epochs, learning_rate)

    # visualize_conv1_features(inferw_net, data_loader, device)

    # Visualize with t-SNE
    visualize_with_tsne(inferw_net, data_loader, device)


def draw_confusion_matrix(predicted_labels, real_labels):
    # Confusion Matrix for GMVAE
    # print(predicted_labels)
    # print(real_labels)
    with torch.no_grad():
        predicted_labels = torch.cat(predicted_labels).cpu().numpy().flatten()
        confusion = confusion_matrix(real_labels, predicted_labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        plt.show()

def visualize_conv1_features(model, data_loader, device, num_channels=10):
    model.eval()

    # Get one batch of data
    inputs, _ = next(iter(data_loader))
    inputs = inputs.to(device)

    # Perform forward pass to get conv1 features
    conv1_features = model.get_conv1_features(inputs)

    # Choose the first 'num_channels' channels for visualization
    conv1_features = conv1_features[:, :num_channels]

    # Reshape and transpose conv1_features for visualization
    conv1_features = conv1_features.permute(0, 2, 3, 1)

    # Create a grid of images for visualization
    num_rows = int(np.ceil(num_channels / 5))
    plt.figure(figsize=(15, 3 * num_rows))
    for i in range(num_channels):
        plt.subplot(num_rows, 5, i + 1)
        plt.imshow(conv1_features[0, :, :, i].cpu().detach().numpy(), cmap='gray')
        plt.title(f'Channel {i + 1}')
        plt.axis('off')
    plt.suptitle('Conv1 Features Visualization')
    plt.show()


def visualize_with_tsne(model, data_loader, device):
    model.eval()

    # Collect all the samples' fc_output and true labels
    fc_outputs = []
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            fc_output, _, pred_label = model(inputs)
            fc_outputs.append(fc_output.cpu().detach().numpy())
            pred_labels.append(torch.topk(pred_label, 1)[1].squeeze(1))
            true_labels.extend(labels.cpu().detach().numpy())
    fc_outputs = np.concatenate(fc_outputs)
    draw_confusion_matrix(pred_labels, true_labels)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=0, init='pca', learning_rate='auto')
    fc_output_tsne = tsne.fit_transform(fc_outputs)

    # Visualize with true labels as colors
    plt.figure(figsize=(10, 8))
    plt.scatter(fc_output_tsne[:, 0], fc_output_tsne[:, 1], s=5, cmap='rainbow', c=true_labels)
    plt.colorbar()
    plt.title('t-SNE Visualization of FC Output with True Labels')
    plt.show()


class InferwNet(nn.Module):
    def __init__(self, image_size, num_clusters):
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
            nn.Linear(64 * 7 * 7, 10)
        )
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, 10))
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


    def get_conv1_features(self, x):
        # This function returns the output of the first convolutional layer
        x = x.view(-1, 1, 28, 28)
        conv_output = self.conv_layers[0](x)
        conv_output = self.conv_layers[1](conv_output)
        conv_output = self.conv_layers[2](conv_output)
        conv_output = self.conv_layers[3](conv_output)
        conv_output = self.conv_layers[4](conv_output)
        return conv_output

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)  # Flatten the tensor
        fc_output = self.fc_layers(features)
        # print(fc_output.cpu().detach().numpy())

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
        loss = 100 * torch.mean(torch.sum(gumbel_w * distances, dim=1))
        print(f"kmeans_loss {loss}")
        return loss

    def inter_cluster_variance(self):
        loss = 1000 * torch.var(self.cluster_centers, dim=0).sum()
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

        loss = 5000 * torch.stack(cluster_variances).mean()
        print(f"intra_loss {loss}")
        return loss

    def inferw_loss(self, features, gumbel_w):
        return self.kmeans_loss(features, gumbel_w) -\
            self.inter_cluster_variance() + self.intra_cluster_variance(features, gumbel_w)

def train_inferw_net(inferw_net, data_loader, num_epochs, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = optim.Adam(inferw_net.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0.0

        for inputs, _ in data_loader:
            inputs = inputs.to(device)

            features, prob, w = inferw_net(inputs)

            inferw_loss = inferw_net.inferw_loss(features, w)

            optimizer.zero_grad()
            inferw_loss.backward()
            optimizer.step()

            total_loss += inferw_loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}")

if __name__ == "__main__":
    main()
