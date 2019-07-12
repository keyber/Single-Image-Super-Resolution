import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.transforms.functional

from model_generator_progressive import GeneratorProgresiveBase, GeneratorSuffix
from model_generator import Generator
from model_discriminator import Discriminator
import model_content_extractor


# commande à utiliser pour lancer le script en restreignant les GPU visibles
#CUDA_VISIBLE_DEVICES=1,2 python3 train.py

progressive = 0

# augmente deux fois la résolution en retournant G(G(x))
forward_twice = 0

# false: x4 pixels   -  true: x16
scale_twice = 2
print("forward_twice", forward_twice, "scale_twice", scale_twice, "progressive", progressive)

# content loss sur les images basse résolution comme dans AmbientGAN
content_loss_on_lr = False

#root directory for trained models
write_root = "/local/beroukhim/srgan_trained/"
print("écriture dans", write_root)

# dataset to use and root directory
dataset_name, dataroot = "celeba", "/local/beroukhim/data/celeba"
# dataset_name, dataroot = "flickr", "/local/beroukhim/data/flickr/images"
# dataset_name, dataroot = "mnist" , "/local/beroukhim/data/mnist"

# affiche les reconstructions à la fin de chaque epoch
plot_training = True # fait planter si n'arrive pas à afficher
if plot_training:
    print("PLOT TRAINING PEUT FAIRE PLANTER LE CODE SI LE SERVEUR X EST INACCESSIBLE")

plot_first = True  # pas encore implémenté

# Learning rate for optimizers
lr = 1e-4

# normalise losses before backward
normalized_gradient = False # plante quand erreur D nulle

# Batch size during training
batch_size = 1
n_batch = 10

# Number of training epochs
num_epochs = 2

# noinspection PyShadowingNames
def gen_modules():
    # Create the generator and discriminator
    if progressive:
        net_g = GeneratorProgresiveBase(n_blocks=16, n_features=64)
        net_g = GeneratorSuffix(net_g, n_features=64)
    else:
        net_g = Generator(list_scales=[2, 2], n_blocks=16, n_features_block=64, n_features_last=256, forward_twice=forward_twice, input_channels=image_size_lr[0])
    
    net_d = Discriminator(image_size_hr, list_n_features=[64, 64, 128, 128, 256, 256, 512, 512],
                           list_stride=[1, 2, 1, 2, 1, 2, 1, 2])
    
    try:
        path = input("entrer le chemin de sauvegarde du réseau à charger:\n")
        checkpoint = torch.load(path)
        net_g.load_state_dict(checkpoint['net_g'])
        net_d.load_state_dict(checkpoint['net_d'])
        print("lecture réussie")
    except Exception as e:
        path = None
        print("lecture échouée", e)
    
    if progressive == 2:
        net_g.beginning[0].requires_grad = False
        net_g = GeneratorSuffix(net_g.beginning, n_features=16)
    
    # create a feature extractor
    identity = model_content_extractor.identity()
    if image_size_lr[0] == 1:
        net_content_extractor = identity
    else:
        net_content_extractor = model_content_extractor.MaskedVGG(0b01111)
    
    # Initialize BCELoss function  #binary cross entropy
    criterion = torch.nn.BCELoss()
    
    return net_g, net_d, identity, net_content_extractor, criterion, path

# noinspection PyShadowingNames
def gen_losses(net_content_extractor, identity):
    n = num_epochs
    n_g = 1, n
    n_d = 1, n
    n_content = 1, n
    n_identity = 0, 1
    
    # noinspection PyShadowingNames
    def loss_weight_adv_g(i):
        if n_g[0] <= i < n_g[1]:
            return 5e-2
        return 0
    
    # noinspection PyShadowingNames
    def loss_weight_adv_d(i):
        if n_d[0] <= i < n_d[1]:
            return 1.0
        return 0
    
    # noinspection PyShadowingNames
    def loss_weight_cont(i):
        cont = n_content[0] <= i < n_content[1]
        iden = n_identity[0] <= i < n_identity[1]
        assert not cont or not iden
        
        if cont:
            return 1.0, net_content_extractor
        
        if iden:
            return 10.0, identity
        
        return 0, None
    
    return loss_weight_adv_g, loss_weight_adv_d, loss_weight_cont

# noinspection PyShadowingNames,PyTypeChecker
def gen_scheduler(optimizerG, optimizerD):
    ratio_lr = .1  # ratio des lr entre iter_0 et max_iter
    # f ^ max_iter = ratio_lr   =>   f = ratio_lr ^ 1/max_iter
    f = ratio_lr ** (1 / (n_batch * num_epochs))
    schedulerG = torch.optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda=lambda iteration: f ** iteration)
    schedulerD = torch.optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda=lambda iteration: f ** iteration)
    return schedulerG, schedulerD

# noinspection PyShadowingNames
def gen_label(device):
    # Establish convention for real and fake labels during training
    real_label = torch.full((batch_size,), 1.0, device=device)
    real_label_reduced = torch.full((batch_size,), .9, device=device)
    fake_label = torch.full((batch_size,), .0, device=device)
    return real_label, real_label_reduced, fake_label

def gen_seed():
    # Set random seem for reproducibility
    # seed = 999
    seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)

# noinspection PyShadowingNames
def gen_dataset():
    if dataset_name == 'celeba':
        # Spatial size of training images. All images will be resized to this size using a transformer.
        image_size_hr = (3, 128, 128)
        if scale_twice or forward_twice or progressive==2:
            image_size_lr = (3, 32, 32)
        else:
            image_size_lr = (3, 64, 64)
    elif dataset_name == 'flickr':
        image_size_hr = (3, 128, 128)
        if scale_twice or forward_twice or progressive==2:
            image_size_lr = (3, 32, 32)
        else:
            image_size_lr = (3, 64, 64)
    elif dataset_name == 'mnist':
        image_size_hr = (1, 28, 28)
        if scale_twice or forward_twice or progressive==2:
            image_size_lr = (1, 7, 7)  # visuellement difficile
        else:
            image_size_lr = (1, 14, 14)
    else:
        raise FileNotFoundError

    # We can use an image folder dataset the way we have it setup.
    # Create the datasets and the dataloaders
    if dataset_name == 'celeba':
        dataset_hr = dset.ImageFolder(root=dataroot,
                                      transform=transforms.Compose([
                                          transforms.Resize(image_size_hr[1:]),
                                          transforms.ToTensor(),
                                          transforms.Normalize((.5, .5, .5), (.5, .5, .5))]))
    elif dataset_name == 'flickr':
        dataset_hr = dset.ImageFolder(root=dataroot,
                                      transform=transforms.Compose([
                                          transforms.CenterCrop((image_size_hr[1] * 2, image_size_hr[2] * 2)),
                                          transforms.Resize(image_size_hr[1:]),
                                          transforms.ToTensor(),
                                          transforms.Normalize((.5, .5, .5), (.5, .5, .5))]))
    elif dataset_name == 'mnist':
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
    return image_size_lr, image_size_hr, dataloader_hr

# noinspection PyShadowingNames
def gen_device(net_g, net_d, net_content_extractor):
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = torch.cuda.device_count()
    
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
    
    return device, net_g, net_d, net_content_extractor

# noinspection PyShadowingNames
def gen_optimizers(checkpoint_path):
    optimizerG = optim.Adam(net_g.parameters(), lr=lr, betas=(.9, 0.999))
    optimizerD = optim.Adam(net_d.parameters(), lr=lr, betas=(.9, 0.999))
    
    if checkpoint_path is not None:
        try:
            checkpoint = torch.load(checkpoint_path)
            optimizerG.load_state_dict(checkpoint['opti_g'])
            optimizerD.load_state_dict(checkpoint['opti_d'])
        except Exception as e:
            print("erreur chargement optimizers", e)
            pass
        
    return optimizerG, optimizerD

gen_seed()
image_size_lr, image_size_hr, dataloader_hr =                                gen_dataset()
if n_batch == -1:
    n_batch = len(dataloader_hr)
net_g, net_d, identity, net_content_extractor, criterion, checkpoint_path =  gen_modules()
device, net_g, net_d, net_content_extractor =                                gen_device(net_g, net_d, net_content_extractor)
optimizerG, optimizerD =                                                     gen_optimizers(checkpoint_path)
loss_weight_adv_g, loss_weight_adv_d, loss_weight_cont =                     gen_losses(net_content_extractor, identity)
schedulerG, schedulerD =                                                     gen_scheduler(optimizerG, optimizerD)
real_label, real_label_reduced, fake_label =                                 gen_label(device)

