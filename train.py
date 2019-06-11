from __future__ import print_function
import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time
from model_generator import Generator
from model_discriminator import Discriminator
import model_content_extractor
import torchvision
import os

# commande à utiliser pour lancer le script en restreignant les GPU visibles
#CUDA_VISIBLE_DEVICES=1,2 python3 train.py

# Set random seem for reproducibility
# seed = 999
seed = random.randint(1, 10000)
print("Random Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

# dataset to use
use_mnist = False

# false: x4 pixels   -  true: x16
scale_twice = True

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

#root directory for trained models
write_root = "/local/beroukhim/srgan_trained/"

# Root directory for dataset
if use_mnist:
    dataroot = "/local/beroukhim/data/mnist"
else:
    dataroot = "/local/beroukhim/data/celeba"

if use_mnist:
    # Spatial size of training images. All images will be resized to this size using a transformer.
    image_size_hr = (1, 28, 28)
    if scale_twice:
        image_size_lr = (1, 7, 7) # visuellement difficile
    else:
        image_size_lr = (1, 14, 14)
else:
    image_size_hr = (3, 128, 128)
    if scale_twice:
        image_size_lr = (3, 32, 32)
    else:
        image_size_lr = (3,  64,  64)

plot_training = False

# Learning rate for optimizers
lr = 1e-4

# Batch size during training
batch_size = 16
n_batch = 2

# Number of training epochs
num_epochs = 2

# Create the generator and discriminator
net_g = Generator(n_blocks=16, n_features=64, scale_twice=scale_twice, input_channels=1 if use_mnist else 3)
net_d = Discriminator(image_size_hr, list_n_features=[64, 64, 128, 128, 256, 256, 512, 512], list_stride=[1, 2, 1, 2, 1, 2, 1, 2])

# create a feature extractor
identity = model_content_extractor.identity()
if image_size_lr[0]==1:
    net_content_extractor = identity
else:
    net_content_extractor = model_content_extractor.MaskedVGG(0b01111)

n0=0
def loss_weight_adv_d(i):
    return i>=n0

def loss_weight_adv_g(i):
    return i>=n0

def loss_weight_cont(i):
    if i < n0:
        return 1, identity
    return 1, net_content_extractor

# Beta1 hyperparam for Adam optimizers
beta1 = 0.9

# utilise le dernier batch comme test (pour l'affichage)
test_last_batch = True

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0


# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
net_g = net_g.to(device)
net_d = net_d.to(device)
if net_content_extractor is not None:
    net_content_extractor = net_content_extractor.to(device)

if (device.type == 'cuda') and (ngpu > 1):
    net_g = nn.DataParallel(net_g, list(range(ngpu)))
    net_d = nn.DataParallel(net_d, list(range(ngpu)))
    if net_content_extractor is not None:
        net_content_extractor = nn.DataParallel(net_content_extractor, list(range(ngpu)))

# Setup Adam optimizers for both G and D
optimizerG = optim.Adam(net_g.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(net_d.parameters(), lr=lr, betas=(beta1, 0.999))

def main():
    # Initialize BCELoss function  #binary cross entropy
    criterion = nn.BCELoss()
    
    
    # We can use an image folder dataset the way we have it setup.
    # Create the datasets and the dataloaders
    if use_mnist:
        dataset_hr = torchvision.datasets.MNIST(dataroot, train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           transforms.Resize(image_size_hr[1:]),
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize((0.5,), (0.5,)),
                                           #torchvision.transforms.Normalize((0.1307,), (0.3081,)), #mean std
                                       ]))
        
        dataset_lr = torchvision.datasets.MNIST(dataroot, train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           transforms.Resize(image_size_lr[1:]),
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize((0.5,), (0.5,)),
                                       ]))
    else:
        dataset_hr = dset.ImageFolder(root=dataroot,
                                      transform=transforms.Compose([
                                       transforms.Resize(image_size_hr[1:]),
                                       transforms.CenterCrop(image_size_hr[1:]),
                                       transforms.ToTensor(),
                                       transforms.Normalize((.5,.5,.5), (.5,.5,.5))]))
        dataset_lr = dset.ImageFolder(root=dataroot,
                                      transform=transforms.Compose([
                                       transforms.Resize(image_size_lr[1:]),
                                       transforms.CenterCrop(image_size_lr[1:]),
                                       transforms.ToTensor(),
                                       transforms.Normalize((.5,.5,.5), (.5,.5,.5))]))
    dataloader_hr = torch.utils.data.DataLoader(dataset_hr, batch_size=batch_size, num_workers=2)
    dataloader_lr = torch.utils.data.DataLoader(dataset_lr, batch_size=batch_size, num_workers=2)
    
    D_losses, G_losses, show_im = train_loop(criterion, dataloader_hr, dataloader_lr)

    # Affichage des résultats
    show_res(D_losses, G_losses, show_im)


def train_loop(criterion, dataloader_hr, dataloader_lr):
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    i_tot = 0
    
    _zero = torch.zeros(1).to(device)
    print_period = max(1, (n_batch if n_batch!=-1 else len(dataloader_hr))//10)
    t = time()
    test_lr, test_hr = None, None
    
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, ((img_lr, _), (img_hr, _)) in enumerate(zip(dataloader_lr, dataloader_hr)):
            img_lr = img_lr.to(device)
            img_hr = img_hr.to(device)
            
            if i == n_batch or i == len(dataloader_hr)-1:
                test_lr, test_hr = img_lr, img_hr
                save_curr_vis(img_list, test_lr, net_g, G_losses, D_losses)
                break
            
            real = img_hr
            
            # Generate fake image batch with G
            fake = net_g(img_lr)
            
            lw_adv_d = loss_weight_adv_d(i_tot)
            if lw_adv_d:
                # Update the Discriminator with adversarial loss
                net_d.zero_grad()
                D_G_z1, D_x, errD = lw_adv_d * adversarial_loss_d(criterion, real, fake)
                optimizerD.step()
            else:
                D_G_z1, D_x, errD  = 0, 0, _zero
            
            # Update the Generator
            net_g.zero_grad()

            # adversarial loss
            lw_adv_g = loss_weight_adv_g(i_tot)
            if lw_adv_g:
                D_G_z2, errG_adv = lw_adv_g * adversarial_loss_g(criterion, fake)
            else:
                D_G_z2, errG_adv = 0, _zero
            
            # content loss
            lw_cont, content_extractor = loss_weight_cont(i_tot)
            if lw_cont and content_extractor is not None :
                errG_cont = lw_cont * content_loss_g(content_extractor, real, fake)
            else:
                errG_cont = _zero
            
            errG = errG_adv + errG_cont
            errG.backward()
            optimizerG.step()
            
            # Output training stats
            if i % print_period == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G_adv: %.4f\tLoss_G_con: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader_hr),
                         errD.item(), errG_adv.item(), errG_cont.item(), D_x, D_G_z1, D_G_z2))
            
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            i_tot += 1
    
    print("train loop in", time() - t)
    return D_losses, G_losses, (test_lr, test_hr, img_list)


def adversarial_loss_d(criterion, real, fake):
    """Update D network: maximize log(D(x)) + log(1 - D(G(z)))"""
    ### Train with all-real batch
    # Forward pass real batch through D
    d_real = net_d(real).view(-1)
    
    # Calculate loss on all-real batch
    label = torch.full((fake.size(0),), real_label, device=device)
    errD_real = criterion(d_real, label)
    
    # Calculate gradients for D in backward pass
    errD_real.backward()
    D_x = d_real.mean().item()
    
    ## Train with all-fake batch
    # Classify all fake batch with D
    d_fake = net_d(fake.detach()).view(-1)
    
    # Calculate D's loss on the all-fake batch
    label.fill_(fake_label)
    errD_fake = criterion(d_fake, label)
    
    # Calculate the gradients for this batch
    errD_fake.backward()
    D_G_z1 = d_fake.mean().item()
    
    # Add the gradients from the all-real and all-fake batches
    errD = errD_real + errD_fake
    return D_G_z1, D_x, errD


def adversarial_loss_g(criterion, fake):
    """Update G network: maximize log(D(G(z)))"""
    # Since we just updated D, perform another forward pass of all-fake batch through D
    output = net_d(fake).view(-1)
    
    # Calculate G's loss based on this output
    label = torch.full((fake.size(0),), real_label, device=device) # fake labels are real for generator cost
    errG = criterion(output, label)
    
    # Calculate gradients for G
    D_G_z2 = output.mean().item()
    return D_G_z2, errG

def content_loss_g(content_extractor, real, fake):
    a = content_extractor(real)
    b = content_extractor(fake)
    return torch.mean(torch.pow(a - b, 2))

import multiprocessing
print_process = None
def save_curr_vis(img_list, img_lr, netG, G_losses, D_losses):
    global print_process
    
    with torch.no_grad():
        fake = netG(img_lr[:16]).detach().cpu()
    img_list.append(vutils.make_grid(fake, padding=2, normalize=True, nrow=4))
    
    def f():
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
        plt.subplot(1,2,2)
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.legend()
        plt.show()
    
    if print_process is not None:
        print_process.terminate()
    
    if plot_training:
        print_process = multiprocessing.Process(target=f)
        print_process.start()


def show_res(D_losses, G_losses, show_im):
    if print_process is not None:
        print_process.terminate()
    
    if not os.path.isdir(write_root):
        os.mkdir(write_root)
        
    i = 0
    path = write_root + str(i)
    while os.path.isfile(path) or os.path.isfile(path+"_ani.mp4"):
        i += 1
        path = write_root + str(i)
        
    torch.save({
        'net_g': net_g.state_dict(),
        'net_d': net_d.state_dict(),
        'opti_g': optimizerG.state_dict(),
        'opti_d': optimizerD.state_dict()
    }, path)
    
    print("réseau sauvegardé dans le fichier", path)

    if not plot_training:
        # attend que l'utilisateur soit là pour créer des figures
        input("modèle écrit, appuyer sur une touche pour afficher")
    
    test_lr, test_hr, img_list = show_im
    
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    # np.transpose inverse les axes pour remettre le channel des couleurs en dernier
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    
    # il faut stocker l'animation dans une variable sinon l'animation plante
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    writer = animation.writers['ffmpeg'](fps=1, bitrate=1800)
    ani.save(path+"_ani.mp4", writer=writer)

    plt.figure(figsize=(15, 8))
    # Plot the LR images
    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.title("LR Images")
    plt.imshow(np.transpose(
        vutils.make_grid(test_lr.to(device)[:16], padding=0, normalize=True, nrow=4).cpu(), (1, 2, 0)))

    # Plot the HR images
    plt.subplot(1, 3, 3)
    plt.axis("off")
    plt.title("HR Images")
    plt.imshow(np.transpose(
        vutils.make_grid(test_hr.to(device)[:16], padding=0, normalize=True, nrow=4).cpu(), (1, 2, 0)))
    
    # Plot the SR from the last epoch
    plt.subplot(1, 3, 2)
    plt.axis("off")
    plt.title("SR Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    main()
