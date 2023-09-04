import torch
import matplotlib.pyplot as plt

def generate_and_compare_samples(model, data_loader, model_param, device, z_dim=20, num_samples=10):
    if model_param == 2:
        random_samples = 3 * torch.rand(num_samples, z_dim).cpu().detach().numpy()
    elif model_param == 3:
        random_samples = torch.bernoulli(torch.full((num_samples, z_dim), 0.5)).cpu().detach().numpy()
    else:
        random_samples = torch.randn(num_samples, z_dim).cpu().detach().numpy()

    random_samples = torch.tensor(random_samples, dtype=torch.float32).to(device)

    if model_param < 4:
        with torch.no_grad():
            generated_samples = model.decode(random_samples)
    else:
        generated_samples = model.generatex(random_samples)

    fig, axes = plt.subplots(nrows=1, ncols=num_samples, figsize=(20, 4))
    for i in range(num_samples):
        axes[i].imshow(generated_samples[i].view(28, 28).cpu(), cmap='gray')
        axes[i].set_title('Sample {}'.format(i + 1))
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
