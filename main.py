import os
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torchvision.utils as vutils
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
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Configure GPU or CPU settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set hyperparameters
image_size = 784
h_dim = 400
z_dim = 20
num_epochs = 3
batch_size = 128
learning_rate = 5e-4
model_param = 0
num_clusters = 10
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
              "1、Gaussian; 2、Uniform; 3、Bernoulli; 4、Gaussian Mixture")
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
        #init cluster centers
        model.inferwNet.initialize_cluster_centers(data_loader, num_clusters, device)
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
            if model_param == 4:
                x_rec_raw, _, _ = model.decode(z, res['categorical'])
                out = x_rec_raw.view(-1, 1, 28, 28)
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

    draw_confusion_matrix(model_param, predicted_labels, real_labels)

    # test_gmvae_w(model, x, y, sample_dir)

    # Perform t-SNE on the latent variables
    # fig, ax = plt.subplots(1, 3)
    # eval_label = np.load(eval_dir + "/eval_label.npy")
    # eval_data = np.load(eval_dir + "/eval_data.npy")
    # plotdistribution(eval_label, eval_data, ax)
    #
    # # Display reconst-1 and reconst-15 images
    # image_1 = mpimg.imread(sample_dir + '/reconst-1.png')
    # plt.subplot(1, 3, 2)
    # ax[1].imshow(image_1)
    # ax[1].set_axis_off()
    #
    # image_15 = mpimg.imread(sample_dir + '/reconst-15.png')
    # plt.subplot(1, 3, 3)
    # ax[2].imshow(image_15)
    # ax[2].set_axis_off()
    # plt.show()


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

def draw_confusion_matrix(model_param, predicted_labels, real_labels):
    # Confusion Matrix for GMVAE
    print(predicted_labels)
    print(real_labels)
    if model_param == 4:
        with torch.no_grad():
            real_labels = torch.cat(real_labels).cpu().numpy().flatten()
            predicted_labels = torch.cat(predicted_labels).cpu().numpy().flatten()
            # create a dictionary to store the real label list for each index
            # label_mapping = {}
            # for pred, real in zip(predicted_labels, real_labels):
            #     if pred not in label_mapping:
            #         label_mapping[pred] = [real]
            #     else:
            #         label_mapping[pred].append(real)
            # # Find the mode of the real label corresponding to each index
            # predicted_indices_mapping = {}
            # for pred, real_list in label_mapping.items():
            #     predicted_indices_mapping[pred] = np.argmax(np.bincount(real_list))
            # print(predicted_indices_mapping)

            confusion = confusion_matrix(real_labels, predicted_labels)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('True Labels')
            ax.set_title('Confusion Matrix')
            plt.show()


def test_gmvae_w(model, x, real_labels, sample_dir):
    x_concat = torch.zeros(x.size(0), 1, 28, 28 * 11)
    # Filter images by digit type
    digit_indices = torch.nonzero((real_labels == 5)).flatten()
    digit_images = x[digit_indices]
    # Choose an image from the dataset
    image = digit_images[0].unsqueeze(0)  # Select the first image and add a batch dimension
    x_concat[:, :, :, 0:28] = image.view(-1, 1, 28, 28)
    #test features of w
    for i in range(10):
        vector = torch.zeros(1, 10).to(device)
        vector[:, i] = 1
        mu, var, z = model.inferz(image, vector)
        x_rec = model.generatex(z)
        x_concat[:, :, :, (i + 1) * 28 : (i + 2) * 28] = x_rec.view(-1, 1, 28, 28)

    # Display input and reconstructed images
    vutils.save_image(x_concat, os.path.join(sample_dir, 'testw_reconst.png'))
    plt.imshow(x_concat[0, 0].cpu().detach().numpy(), cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()

