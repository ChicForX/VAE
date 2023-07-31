import os
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
import numpy as np
import warnings
import sys
import gaussian
import uniform
import bernoulli
import gm
import inferw_pretrain
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from scipy.stats import mode
from collections import defaultdict

# Configure GPU or CPU settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set hyperparameters
image_size = 784
h_dim = 400
z_dim = 20
num_epochs = 16
batch_size = 128
learning_rate = 1e-3
model_param = 0
num_classes = 10
# Get the dataset
dataset = torchvision.datasets.MNIST(root='./data',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)

# Load data by batch size and shuffle randomly
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=True)


def main():
    # Parsing Command Line Parameters
    if len(sys.argv) < 2:
        print("Please input the distribution for the latent variables as argv[1]："
              "1、Gaussian; 2、Uniform; 3、Bernoulli; 4、Gaussian Mixture(Unsupervised); "
              "5、Gaussian Mixture(Semi-supervised)")
        return

    model_param = int(sys.argv[1])
    # Create directories to save generated images, evaluation
    sample_dir = 'samples'
    eval_dir = 'result'
    # 1-gaussian, 2-uniform, 3-bernoulli, 4-mixtured gaussian
    if model_param == 1:
        model = gaussian.VAE().to(device)
        sample_dir += '_gaussian'
        eval_dir += '_gaussian'
    elif model_param == 2:
        model = uniform.VAE().to(device)
        sample_dir += '_uniform'
        eval_dir += '_uniform'
    elif model_param == 3:
        model = bernoulli.VAE().to(device)
        sample_dir += '_bernoulli'
        eval_dir += '_bernoulli'
    elif model_param == 4:
        model = gm.VAE().to(device)
        sample_dir += '_gaussianmixture'
        eval_dir += '_gaussianmixture'
        #for confusion matrix
        predicted_labels = []
        real_labels = []
    elif model_param == 5:
        model = gm.VAE().to(device)
        sample_dir += '_gaussianmixture'
        eval_dir += '_gaussianmixture'
        # for confusion matrix
        predicted_labels = []
        real_labels = []
        model = inferw_pretrain.main(data_loader, model, image_size, device)
        accuracy_pretrain = inferw_pretrain.accuracy_after_pretrain(model.inferwNet, data_loader, image_size, device)
    else:
        print("Invalid parameter!")
        return
    print(model)

    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Training loop
    for epoch in range(num_epochs):
        y_np = []
        z_np = []
        for i, (x, y) in enumerate(data_loader):
            # forward propagation
            x = x.to(device).view(-1, image_size)  # batch_size*1*28*28 ---->batch_size*image_size  where image_size=1*28*28=784
            res = model(x)
            x_reconst = res['x_rec']
            z = res['sample']

            # Record the output of latentspace
            y_cpu = y.cpu().detach().numpy()
            z_cpu = z.cpu().detach().numpy()
            y_np.extend(y_cpu)
            z_np.extend(z_cpu)  # batch*20

            # Calculate reconstruction loss and KL divergence
            if model_param == 4 or model_param == 5:
                reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
                reconst_loss = reconst_loss/batch_size
                kl_divergence = 3.5 * model.kl_divergence(res)
            else:
                reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
                kl_divergence = model.kl_divergence(res)

            # Backpropagation and Optimization
            loss = reconst_loss + kl_divergence
            # Clear the residual update parameter values from the previous step
            optimizer.zero_grad()
            # Error backpropagation, calculate parameter values
            loss.backward()
            # update parameter values
            optimizer.step()
            # print the loss
            if (i + 1) % 10 == 0:
                print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, len(data_loader), reconst_loss.item(),
                              kl_divergence.item()))
                if not os.path.exists(eval_dir):
                    os.mkdir(eval_dir)
                if os.path.exists(eval_dir + "/eval_label.npy"):
                    os.remove(eval_dir + "/eval_label.npy")
                if os.path.exists(eval_dir + "/eval_data.npy"):
                    os.remove(eval_dir + "/eval_data.npy")
                np.save(eval_dir + "/eval_label.npy", y_np)
                np.save(eval_dir + "/eval_data.npy", z_np)

        with torch.no_grad():
            # saving sampled values
            z = torch.randn(batch_size, z_dim).to(device)  # z: batch_size * z_dim = 128*20
            # Decode and output the random number z
            if model_param == 4 or model_param == 5:
                x_rec_raw, _, _ = model.decode(z, res['categorical'])
                out = x_rec_raw.view(-1, 1, 28, 28)
                if epoch > num_epochs - 3:
                    predicted_labels.append(torch.topk(res['categorical'], 1)[1].squeeze(1))
                    real_labels.append(y)
            else:
                out = model.decode(z).view(-1, 1, 28, 28)
            save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch + 1)))

            # save reconstruction values
            # x: batch_size*748, forward propagation, obtaining reconstruction value out
            out = model(x)
            # Splice input and output together, output and save
            # batch_size*1*28*（28+28）=batch_size*1*28*56
            x_concat = torch.cat([x.view(-1, 1, 28, 28), out['x_rec'].view(-1, 1, 28, 28)], dim=3)
            save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch + 1)))

    if model_param == 4 or model_param == 5:
        # 10 classes
        label_mappings = get_pred_real_label_mapping(predicted_labels, real_labels)
        print(label_mappings)
        print(f"Validation Accuracy After Pretrain: {accuracy_pretrain:.4f}")
        draw_confusion_matrix(predicted_labels, real_labels, label_mappings)
        test_gmvae_w(model, x, y, sample_dir, label_mappings)

        # 4 classes
        # target_mapping = {0: 0, 6: 0, 8: 0, 1: 1, 7: 1, 2: 2,
        #                   3: 2, 5: 2, 4: 3, 9: 3}
        # print(f"Validation Accuracy After Pretrain: {accuracy_pretrain:.4f}")
        # draw_confusion_matrix(predicted_labels, real_labels, target_mapping)
        # label_mappings = {0:0, 1:1, 2:2, 3:3}
        # test_gmvae_w(model, x, y, sample_dir, label_mappings)

    # Perform t-SNE on the latent variables
    fig, ax = plt.subplots(1, 3)
    eval_label = np.load(eval_dir + "/eval_label.npy")
    eval_data = np.load(eval_dir + "/eval_data.npy")
    plotdistribution(eval_label, eval_data, ax)

    # Display reconst-1 and reconst-15 images
    image_1 = mpimg.imread(sample_dir + '/reconst-1.png')
    plt.subplot(1, 3, 2)
    ax[1].imshow(image_1)
    ax[1].set_axis_off()

    image_15 = mpimg.imread(sample_dir + '/reconst-15.png')
    plt.subplot(1, 3, 3)
    ax[2].imshow(image_15)
    ax[2].set_axis_off()
    plt.show()


def plotdistribution(Label, Mat, ax):
    warnings.filterwarnings('ignore', category=FutureWarning)
    tsne = TSNE(n_components=2, random_state=0)
    Mat = tsne.fit_transform(Mat[:])

    x = Mat[:, 0]
    y = Mat[:, 1]
    # map_size = {0: 5, 1: 5}
    # size = list(map(lambda x: map_size[x], Label))
    map_color = {0: 'r', 1: 'g',2:'b',3:'y',4:'k',5:'m',6:'c',7:'pink',8:'grey',9:'blueviolet'}
    color = list(map(lambda x: map_color[x], Label))
    # error occurs because the marker parameter does not support lists
    # map_marker = {-1: 'o', 1: 'v'}
    # markers = list(map(lambda x: map_marker[x], Label))
    #  plt.scatter(np.array(x), np.array(y), s=size, c=color, marker=markers)
    ax[0].scatter(np.array(x), np.array(y), s=5, c=color, marker='o')  # The scatter function only supports array type data
    ax[0].set_axis_on()

    # add labels
    legend_elements = []
    for label, color in map_color.items():
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=5, label=label))
    ax[0].legend(handles=legend_elements, title='Label', loc='upper right', handlelength=0.8, handleheight=0.8)



def find_pred_true_mapping(predicted_labels, real_labels):
    label_mappings = {}
    label_rates = {}
    for i in np.unique(real_labels):
        target_value = i
        indices = np.where(real_labels == target_value)[0]

        corresponding_values = predicted_labels[indices]

        mode_result = mode(corresponding_values, keepdims=True)
        most_common_value = mode_result.mode[0]
        label_rate = np.count_nonzero(corresponding_values == most_common_value) / len(corresponding_values)

        if (most_common_value in label_rates and label_rates[most_common_value] is not None \
                and label_rate > label_rates[most_common_value]) or \
                (most_common_value not in label_rates or label_rates[most_common_value] is None):
            label_rates[most_common_value] = label_rate
            # mapping: pred -> real
            label_mappings[most_common_value] = i

    return label_mappings

def check_duplicate_labels(predicted_labels, real_labels, label_mappings):

    while True:
        mapping_values = set(label_mappings.values())
        mapping_keys = set(label_mappings.keys())
        left_real_labels = set(real_labels) - mapping_values
        if len(left_real_labels) <= 0:
            break
        elif len(left_real_labels) == 1:
            left_pred_labels = set(predicted_labels) - mapping_keys
            label_mappings[left_pred_labels.pop()] = left_real_labels.pop()
        else:
            new_label_idx1 = np.array([i for i in range(len(real_labels)) if real_labels[i] in left_real_labels])
            new_label_idx2 = np.array([i for i in range(len(predicted_labels)) if predicted_labels[i] not in mapping_keys])
            new_label_idx = np.intersect1d(new_label_idx1, new_label_idx2)
            if len(new_label_idx) > 0:
                left_label_mappings = find_pred_true_mapping(predicted_labels[new_label_idx], real_labels[new_label_idx])
                label_mappings.update(left_label_mappings)
            else:
                left_pred_labels = sorted(set(predicted_labels) - mapping_keys)
                left_real_labels = sorted(left_real_labels)
                for i, pred_label in enumerate(left_pred_labels):
                    label_mappings[pred_label] = left_real_labels[i]


    return label_mappings

def replace_confusion_elements(value, label_mappings):
    return label_mappings.get(value, value)

def get_pred_real_label_mapping(predicted_labels, real_labels):
    real_labels = torch.cat(real_labels).cpu().numpy().flatten()
    predicted_labels = torch.cat(predicted_labels).cpu().numpy().flatten()

    label_mappings = find_pred_true_mapping(predicted_labels, real_labels)
    label_mappings = check_duplicate_labels(predicted_labels, real_labels, label_mappings)
    return label_mappings

def draw_confusion_matrix(predicted_labels, real_labels, label_mappings):
    # Confusion Matrix for GMVAE
    # print(predicted_labels)
    # print(real_labels)

    with torch.no_grad():
        real_labels = torch.cat(real_labels).cpu().numpy().flatten() # 10 classes
        # real_labels = [label_mappings[label.item()] for labels in real_labels for label in labels]# 4 classes

        predicted_labels = torch.cat(predicted_labels).cpu().numpy().flatten()

        # 10 classes
        vectorized_replace = np.vectorize(replace_confusion_elements)
        adjusted_pred_labels = vectorized_replace(predicted_labels, label_mappings)
        confusion = confusion_matrix(real_labels, adjusted_pred_labels)

        # 4 classes
        # confusion = confusion_matrix(real_labels, predicted_labels)

        # Calculate the accuracy of prediction
        # accuracy = accuracy_score(real_labels, adjusted_pred_labels)
        accuracy = accuracy_score(real_labels, predicted_labels)
        print(f"Accuracy of prediction after GMVAE:    {accuracy:.4f}")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        plt.show()

def test_gmvae_w(model, x, real_labels, sample_dir, label_mappings):
    images_list = []
    # Filter images by digit type
    digit_indices = torch.nonzero((real_labels == 5)).flatten()
    digit_images = x[digit_indices]
    # Choose an image from the dataset
    image = digit_images[0].unsqueeze(0)  # Select the first image and add a batch dimension

    # different weight
    for w_weight in range(1, 11, 1):
        x_concat = torch.zeros(x.size(0), 1, 28, 28 * (1+num_classes))
        x_concat[:, :, :, 0:28] = image.view(-1, 1, 28, 28)
        # different category
        for i in range(num_classes):
            vector = torch.zeros(1, num_classes).to(device)
            j = next((key for key, value in label_mappings.items() if value == i), None)
            vector[:, j] = w_weight
            mu, var, z = model.inferz(image, vector)
            x_rec = model.generatex(z)
            x_concat[:, :, :, (i + 1) * 28: (i + 2) * 28] = x_rec.view(-1, 1, 28, 28)

        images_list.append(x_concat)

    vertical_concatenated_image = torch.cat(images_list, dim=2)  # Concatenate along dimension 2
    # Display input and reconstructed images
    save_image(vertical_concatenated_image, os.path.join(sample_dir, 'testw_reconst.png'), nrow=num_classes)
    plt.imshow(vertical_concatenated_image[0, 0].cpu().detach().numpy(), cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()

