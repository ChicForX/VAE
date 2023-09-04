import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Load MNIST dataset
dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)

# Create a list to store the indices of the selected samples
selected_indices = []

# Select 10 random digits
for digit in range(10):
    digit_indices = np.where(dataset.targets == digit)[0]
    selected_indices.extend(np.random.choice(digit_indices, size=10, replace=False))

# Load the selected samples and labels
selected_samples = torch.stack([dataset[i][0] for i in selected_indices])
selected_labels = [dataset[i][1] for i in selected_indices]

# Create a grid of images
grid = torchvision.utils.make_grid(selected_samples, nrow=10)

# Plot the grid of images
plt.figure(figsize=(10, 10))
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.title('Randomly Selected MNIST Samples')
plt.show()
