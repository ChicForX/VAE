import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset

# num_classes = 4
# num_samples_per_class = 7

num_classes = 10
num_samples_per_class = 5


def load_mnist_class(train_dataset, class_label, num_samples, is_train=True):
    class_indices = [i for i in range(len(train_dataset.targets)) if train_dataset.targets[i] == class_label]
    selected_indices = class_indices[:num_samples]
    subset_dataset = Subset(train_dataset, selected_indices)

    if is_train:
        return subset_dataset
    else:
        # use the remaining data to create a subset of the validation set
        remaining_indices = list(set(range(len(train_dataset))) - set(selected_indices))
        num_validation_samples = 200

        validation_indices = remaining_indices[:num_validation_samples]
        validation_dataset = Subset(train_dataset, validation_indices)

        return validation_dataset

def supervised_pretraining(model, data_loader, image_size, device, learning_rate=0.001, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device).view(-1, image_size)
            targets = get_target_mapping(num_classes, targets).to(device)
            _, logits, _ = model(inputs, image_size)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")


def main(data_loader, model, image_size, device):
    all_pretrain_data = []

    for class_label in range(10):
        print(f"Loading data for class: {class_label}")
        pretrain_data = load_mnist_class(data_loader.dataset, class_label, num_samples=num_samples_per_class)
        all_pretrain_data.append(pretrain_data)

    # Combine all data into a single dataset
    combined_pretrain_dataset = ConcatDataset(all_pretrain_data)
    pretrain_data_loader = DataLoader(combined_pretrain_dataset, batch_size=num_samples_per_class, shuffle=True)
    # pretrain
    supervised_pretraining(model.inferwNet, pretrain_data_loader, image_size, device)


    return model

def accuracy_after_pretrain(model, data_loader, image_size, device):
    all_validate_data = []
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        # create validation data by remained data
        for class_label in range(num_classes):
            validation_data = load_mnist_class(data_loader.dataset, class_label,
                                               num_samples=num_samples_per_class, is_train=False)
            all_validate_data.append(validation_data)

        #calculate the accuracy before GMVAE
        combined_validate_dataset = ConcatDataset(all_validate_data)
        validate_data_loader = DataLoader(combined_validate_dataset, batch_size=1000, shuffle=True)
        for inputs, targets in validate_data_loader:
            inputs = inputs.to(device).view(-1, image_size)
            targets = get_target_mapping(num_classes, targets).to(device)
            logits, _, _ = model(inputs, image_size)
            _, predicted_labels = torch.max(logits, 1)
            correct_predictions += (predicted_labels == targets).sum().item()
            total_samples += len(targets)

    accuracy = correct_predictions / total_samples
    return accuracy

def get_target_mapping(num_classes, targets):
    if num_classes == 4:
        # target_mapping = {0: 0, 6: 0, 8: 0, 1: 1, 7: 1, 2: 2,
        #                   3: 2, 5: 2, 4: 3, 9: 3}
        target_mapping = {0: 0, 6: 1, 8: 3, 1: 3, 7: 0, 2: 1,
                          3: 2, 5: 3, 4: 2, 9: 1}
        targets = torch.tensor([target_mapping[target.item()] for target in targets], dtype=torch.long)
    return targets


