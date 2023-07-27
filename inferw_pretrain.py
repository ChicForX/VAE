import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset

def load_mnist_class(train_dataset, class_label, num_samples=20):
    class_indices = [i for i in range(len(train_dataset.targets)) if train_dataset.targets[i] == class_label]
    selected_indices = class_indices[:num_samples]
    subset_dataset = Subset(train_dataset, selected_indices)

    return subset_dataset

def supervised_pretraining(model, data_loader, image_size, device, learning_rate=0.001, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device).view(-1, image_size)
            targets = targets.to(device)
            logits, _, _ = model(inputs, image_size)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

def main(data_loader, model, image_size, device):
    num_classes = 10
    num_samples_per_class = 20

    all_data = []
    for class_label in range(num_classes):
        print(f"Loading data for class: {class_label}")
        pretrain_data = load_mnist_class(data_loader.dataset, class_label, num_samples=num_samples_per_class)
        all_data.append(pretrain_data)

    # Combine all data into a single dataset
    combined_dataset = ConcatDataset(all_data)
    pretrain_data_loader = DataLoader(combined_dataset, batch_size=num_samples_per_class, shuffle=True)

    supervised_pretraining(model.inferwNet, pretrain_data_loader, image_size, device)

    return model

#TODO compate the accuracy of prediction before and after the VAE training