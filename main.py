import os
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.manifold import TSNE
import numpy as np
import warnings
import sys
import gaussian
import uniform
import bernoulli
import gm

# Configure GPU or CPU settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set hyperparameters
image_size = 784
h_dim = 400
z_dim = 20
num_epochs = 15
batch_size = 128
learning_rate = 1e-3

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
    # 1-gaussian, 2-uniform, 3-mixtured gaussian
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
    else:
        print("Invalid parameter!")
        return
    print(model)
    """VAE(
      (fc1): Linear(in_features=784, out_features=400, bias=True)
      (fc2): Linear(in_features=400, out_features=20, bias=True)
      (fc3): Linear(in_features=400, out_features=20, bias=True)
      (fc4): Linear(in_features=20, out_features=400, bias=True)
      (fc5): Linear(in_features=400, out_features=784, bias=True)
    )"""
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
            x_reconst, mu, log_var, z = model(x)  # x:batch_size*748

            # Record the output of latentspace
            y_cpu = y.cpu().detach().numpy()
            z_cpu = z.cpu().detach().numpy()
            y_np.extend(y_cpu)
            z_np.extend(z_cpu)  # batch*20

            # Calculate reconstruction loss and KL divergence
            reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')

            kl_divergence = model.kl_divergence(mu, log_var)

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
            out = model.decode(z).view(-1, 1, 28, 28)
            save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch + 1)))

            # save reconstruction values
            # x: batch_size*748, forward propagation, obtaining reconstruction value out
            out, _, _, _ = model(x)
            # Splice input and output together, output and save
            # batch_size*1*28*（28+28）=batch_size*1*28*56
            x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
            save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch + 1)))

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

if __name__ == "__main__":
    main()




