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
import model_feature_extractor

# Set random seem for reproducibility
# seed = 999
seed = random.randint(1, 10000)
print("Random Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

# Root directory for dataset
dataroot = "/local/beroukhim/data/celeba"

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0

# Spatial size of training images. All images will be resized to this size using a transformer.
image_size_hr = (3, 64, 64)
image_size_lr = (3, 32, 32)  # shapes need to match the scaling

# content loss weight
k = 1e-3

# Learning rate for optimizers
lr = 1e-4

# Batch size during training
batch_size = 16
n_batch = 2

# Number of training epochs
num_epochs = 3

# Create the generator and discriminator
net_g = Generator(n_blocks=4, n_features=32, scale_twice=False)
net_d = Discriminator(image_size_hr, list_n_features=[32, 64, 64], list_stride=[2, 1, 2])

# create a feature extractor
net_feature_extractor = model_feature_extractor.vgg()

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# utilise le premier batch pour l'affichage
test_first_batch = True

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0


# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
net_g = net_g.to(device)
net_d = net_d.to(device)
if (device.type == 'cuda') and (ngpu > 1):
    net_g = nn.DataParallel(net_g, list(range(ngpu)))
    net_d = nn.DataParallel(net_d, list(range(ngpu)))


def main():
    # Initialize BCELoss function  #binary cross entropy
    criterion = nn.BCELoss()
    
    # Setup Adam optimizers for both G and D
    optimizerG = optim.Adam(net_g.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(net_d.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # We can use an image folder dataset the way we have it setup.
    # Create the datasets and the dataloaders
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
    
    D_losses, G_losses, img_list = train_loop(criterion, dataloader_hr, dataloader_lr, optimizerD, optimizerG)

    # Affichage des résultats
    show_res(D_losses, G_losses, dataloader_lr, img_list)


def train_loop(criterion, dataloader_hr, dataloader_lr, optimizerD, optimizerG):
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    print_period = max(1, n_batch//10)
    t = time()
    
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, ((img_lr, _), (img_hr, _)) in enumerate(zip(dataloader_lr, dataloader_hr)):
            if test_first_batch and i == 0:
                save_curr_vis(img_list, img_lr, net_g)
                continue
            
            if i == n_batch:
                break
            
            real = img_hr.to(device)
            
            # Generate fake image batch with G
            fake = net_g(img_lr.to(device))
            
            
            # Update the Discriminator with adversarial loss
            net_d.zero_grad()
            D_G_z1, D_x, errD = adversarial_loss_d(criterion, real, fake)
            optimizerD.step()
            
            
            # Update the Generator
            net_g.zero_grad()
            
            # adversarial loss
            D_G_z2, errG_adv = adversarial_loss_g(criterion, fake)
            
            # content loss
            errG_cont = content_loss_g(real, fake)
            
            errG = errG_adv + k*errG_cont
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
            
    print("train loop in", time() - t)
    return D_losses, G_losses, img_list


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

def content_loss_g(real, fake):
    a = net_feature_extractor(real)
    b = net_feature_extractor(fake)
    return torch.mean(torch.pow(a - b, 2))


def save_curr_vis(img_list, img_lr, netG):
    with torch.no_grad():
        fake = netG(img_lr).detach().cpu()
    img_list.append(vutils.make_grid(fake, padding=2, normalize=True, nrow=4))


def show_res(D_losses, G_losses, dataloader_lr, img_list):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    #np.transpose inverse les axes pour remettre le channel des couleurs en dernier
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    
    # Plot the LR images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("LR Images")
    plt.imshow(np.transpose(
        vutils.make_grid(next(iter(dataloader_lr))[0].to(device)[:16], padding=5, normalize=True, nrow=4).cpu(),
        (1, 2, 0)))
    
    # Plot the SR from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("SR Images")
    plt.imshow(np.transpose(img_list[-1][:16], (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    main()
