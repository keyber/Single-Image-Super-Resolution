import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.transforms.functional

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

# dataset to use & root directory
dataset_name, dataroot = "celeba", "/local/beroukhim/data/celeba"
# dataset_name, dataroot = "flickr", "/local/beroukhim/data/flickr/images"
# dataset_name, dataroot = "mnist" , "/local/beroukhim/data/mnist"

# augmente deux fois la résolution en retournant G(G(x))
forward_twice = 0

# false: x4 pixels   -  true: x16
scale_twice = 3
print("forward_twice", forward_twice, "scale_twice", scale_twice)

# content loss sur les images basse résolution comme dans AmbientGAN
content_loss_on_lr = False

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

#root directory for trained models
write_root = "/local/beroukhim/srgan_trained/"


if dataset_name=='celeba':
    # Spatial size of training images. All images will be resized to this size using a transformer.
    image_size_hr = (3, 128, 128)
    if scale_twice or forward_twice:
        image_size_lr = (3, 32, 32)
    else:
        image_size_lr = (3,  64,  64)
elif dataset_name=='flickr':
    image_size_hr = (3, 128, 128)
    if scale_twice or forward_twice:
        image_size_lr = (3, 32, 32)
    else:
        image_size_lr = (3,  64,  64)
elif dataset_name=='mnist':
    image_size_hr = (1, 28, 28)
    if scale_twice or forward_twice:
        image_size_lr = (1, 7, 7) # visuellement difficile
    else:
        image_size_lr = (1, 14, 14)
else:
    raise FileNotFoundError

plot_training = True

# Learning rate for optimizers
lr = 1e-4

# Batch size during training
batch_size = 16
n_batch = 1000

# Number of training epochs
num_epochs = 5


# We can use an image folder dataset the way we have it setup.
# Create the datasets and the dataloaders
if dataset_name == 'celeba':
    dataset_hr = dset.ImageFolder(root=dataroot,
                                  transform=transforms.Compose([
                                   transforms.Resize(image_size_hr[1:]),
                                   transforms.ToTensor(),
                                   transforms.Normalize((.5,.5,.5), (.5,.5,.5))]))
elif dataset_name == 'flickr':
    dataset_hr = dset.ImageFolder(root=dataroot,
                                  transform=transforms.Compose([
                                   transforms.CenterCrop((image_size_hr[1]*2,image_size_hr[2]*2)),
                                   transforms.Resize(image_size_hr[1:]),
                                   transforms.ToTensor(),
                                   transforms.Normalize((.5,.5,.5), (.5,.5,.5))]))
elif dataset_name=='mnist':
    dataset_hr = torchvision.datasets.MNIST(dataroot, train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       transforms.Resize(image_size_hr[1:]),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.5,), (0.5,)),
                                       #torchvision.transforms.Normalize((0.1307,), (0.3081,)), #mean std
                                   ]))
else:
    raise FileNotFoundError

dataloader_hr = torch.utils.data.DataLoader(dataset_hr, batch_size=batch_size, num_workers=2, drop_last=True)

if n_batch == -1:
    n_batch = len(dataloader_hr)

# Create the generator and discriminator
net_g = Generator(n_blocks=16, n_features=64, forward_twice=forward_twice, scale_twice=scale_twice, input_channels=image_size_lr[0])
net_d = Discriminator(image_size_hr, list_n_features=[64, 64, 128, 128, 256, 256, 512, 512], list_stride=[1, 2, 1, 2, 1, 2, 1, 2])

# create a feature extractor
identity = model_content_extractor.identity()
if image_size_lr[0]==1:
    net_content_extractor = identity
else:
    net_content_extractor = model_content_extractor.MaskedVGG(0b01111)

# Initialize BCELoss function  #binary cross entropy
criterion = torch.nn.BCELoss()

n = num_epochs
n_g        = 0, n
n_d        = 0, n
n_content  = n, n
n_identity = 0, n
n_none     = n, n

def loss_weight_adv_g(i):
    return n_g[0] <= i < n_g[1]

def loss_weight_adv_d(i):
    return n_d[0] <= i < n_d[1]

def loss_weight_cont(i):
    cont = n_content[0] <= i < n_content[1]
    iden = n_identity[0] <= i < n_identity[1]
    assert not cont or not iden
    
    if cont:
        return 1, net_content_extractor
    
    if iden:
        return 1, identity
    
    return 0, None

# Beta1 hyperparam for Adam optimizers
beta1 = 0.9

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
net_g = net_g.to(device)
net_d = net_d.to(device)
if net_content_extractor is not None:
    net_content_extractor = net_content_extractor.to(device)

# Setup Adam optimizers for both G and D
optimizerG = optim.Adam(net_g.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(net_d.parameters(), lr=lr, betas=(beta1, 0.999))

ratio_lr = .1 # ratio des lr entre iter_0 et max_iter
# f ^ max_iter = ratio_lr   =>   f = ratio_lr ^ 1/max_iter
f = ratio_lr ** (1 / (n_batch*num_epochs))
schedulerG = torch.optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda=lambda iteration: f**iteration)
schedulerD = torch.optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda=lambda iteration: f**iteration)

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

if (device.type == 'cuda') and (ngpu > 1):
    net_g = nn.DataParallel(net_g, list(range(ngpu)))
    net_d = nn.DataParallel(net_d, list(range(ngpu)))
    if net_content_extractor is not None:
        net_content_extractor = nn.DataParallel(net_content_extractor, list(range(ngpu)))


# Establish convention for real and fake labels during training
real_label = torch.full((batch_size,), .9, device=device)
fake_label = torch.full((batch_size,), .0, device=device)