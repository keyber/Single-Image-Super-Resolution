import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from model_generator import Generator
from model_discriminator import Discriminator
import model_content_extractor


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
scale_twice = False

# content loss sur les images basse résolution comme dans AmbientGAN
content_loss_on_lr = False

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
n_batch = -1

# Number of training epochs
num_epochs = 3

# Create the generator and discriminator
net_g = Generator(n_blocks=16, n_features=64, scale_twice=scale_twice, input_channels=1 if use_mnist else 3)
net_d = Discriminator(image_size_hr, list_n_features=[64, 64, 128, 128, 256, 256, 512, 512], list_stride=[1, 2, 1, 2, 1, 2, 1, 2])

# create a feature extractor
identity = model_content_extractor.identity()
if image_size_lr[0]==1:
    net_content_extractor = identity
else:
    net_content_extractor = model_content_extractor.MaskedVGG(0b01111)

n0 = 10
def loss_weight_adv_d(i):
    return i>=n0

def loss_weight_adv_g(i):
    return i>=n0

def loss_weight_cont(i):
    if content_loss_on_lr:
        return 1, identity
    else:
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

try:
    path = input("entrer le chemin de sauvegarde du réseau à charger:\n")
    checkpoint = torch.load(path)
    net_g.load_state_dict(checkpoint['net_g'])
    net_d.load_state_dict(checkpoint['net_d'])
    optimizerG.load_state_dict(checkpoint['opti_g'])
    optimizerD.load_state_dict(checkpoint['opti_d'])
    print("lecture réussie")
except Exception as e:
    print("lecture échouée", e)

