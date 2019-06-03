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
import generator
import discriminator


# Set random seem for reproducibility
# manualSeed = 999
manualSeed = random.randint(1, 10000)  # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataroot = "/local/beroukhim/data/celeba"

# Number of workers for dataloader
workers = 2
# Spatial size of training images. All images will be resized to this size using a transformer.
image_size_hr = (3, 64, 64)
image_size_lr = (3, 32, 32)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0


# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0


# Learning rate for optimizers
lr = 1e-4

# Batch size during training
batch_size = 16
n_batch = 1001

# Number of training epochs
num_epochs = 3
test_first_batch = True


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

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


# Create the generator and discriminator
netG = generator.Generator(n_blocks=4, n_features=32).to(device)
netD = discriminator.Discriminator(image_size_hr, list_n_features=[32,64,64], list_stride=[2,1,2]).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
    netD = nn.DataParallel(netD, list(range(ngpu)))

# print(netG)
# print(netD)

# Initialize BCELoss function  #binary cross entropy
criterion = nn.BCELoss()

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
t = time()


print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, ((img_lr, _), (img_hr, _)) in enumerate(zip(dataloader_lr, dataloader_hr)):
        if test_first_batch and i==0:
            with torch.no_grad():
                fake = netG(img_lr).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True, nrow=4))
            continue
        
        if i == n_batch:
            break

        ##### (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) #####
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        data = img_hr.to(device)
        b_size = data.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = netD(data).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate fake image batch with G
        fake = netG(img_lr)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ##### (2) Update G network: maximize log(D(G(z))) #####
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 500 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader_hr),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

print("train loop in", time() - t)

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


# Plot the real images
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("LR Images")
plt.imshow(np.transpose(vutils.make_grid(next(iter(dataloader_lr))[0].to(device)[:16], padding=5, normalize=True, nrow=4).cpu(), (1, 2, 0)))

# Plot the fake images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("SR Images")
plt.imshow(np.transpose(img_list[-1][:16], (1, 2, 0)))
plt.show()