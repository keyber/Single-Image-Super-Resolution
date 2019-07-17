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
import utils
import numpy as np

# commande à utiliser pour lancer le script en restreignant les GPU visibles
#CUDA_VISIBLE_DEVICES=1,2 python3 train.py

list_scales = [2, 2]
scale_factor = np.prod(list_scales)
print("list_scales", list_scales)

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
plot_training = False # fait planter si n'arrive pas à afficher
if plot_training:
    print("PLOT TRAINING PEUT FAIRE PLANTER LE CODE SI LE SERVEUR X EST INACCESSIBLE")

plot_first = True

# Learning rate for optimizers
lr = 1e-4

# normalise losses before backward
normalized_gradient = False # plante quand erreur D nulle

# Batch size during training
batch_size = 16
n_batch = -1

# Number of training epochs
num_epochs = 1

# noinspection PyShadowingNames
def gen_modules(ngpu):
    # Create the generator and discriminator
    net_g = Generator(list_scales=list_scales, n_blocks=16, n_features_block=64, n_features_last=256, input_channels=image_size_lr[0])
    
    net_d = Discriminator(image_size_hr, list_n_features=[64, 64, 128, 128, 256, 256, 512, 512],
                           list_stride=[1, 2, 1, 2, 1, 2, 1, 2])
    
    try:
        path = input("entrer le chemin de sauvegarde du réseau à charger:\n")
        checkpoint = torch.load(path, map_location='cpu')
        net_g.load_state_dict(checkpoint['net_g'], strict=False)
        net_d.load_state_dict(checkpoint['net_d'], strict=False)
        print("lecture réussie")
    except OSError as e:
        path = None
        print("lecture échouée", e)
    
    # net_g.freeze()
    
    # create a feature extractor
    identity = model_content_extractor.identity()
    if image_size_lr[0] == 1:
        net_content_extractor = identity
    else:
        net_content_extractor = model_content_extractor.MaskedVGG(0b01111)
    
    # Initialize BCELoss function  #binary cross entropy
    criterion = torch.nn.BCELoss()
    
    net_g = net_g.to(device)
    net_d = net_d.to(device)
    if net_content_extractor is not None:
        net_content_extractor = net_content_extractor.to(device)
    
    if (device.type == 'cuda') and (ngpu > 1):
        net_g = nn.DataParallel(net_g, list(range(ngpu)))
        net_d = nn.DataParallel(net_d, list(range(ngpu)))
        if net_content_extractor is not None:
            net_content_extractor = nn.DataParallel(net_content_extractor, list(range(ngpu)))

    
    return net_g, net_d, identity, net_content_extractor, criterion, path

# noinspection PyShadowingNames
def gen_losses(net_content_extractor, identity):
    n = num_epochs
    n_g = 0, n
    n_d = 0, n
    n_content = 0, n
    n_identity = 0, 0
    
    # noinspection PyShadowingNames
    def loss_weight_adv_g(i):
        if n_g[0] <= i < n_g[1]:
            if content_loss_on_lr:
                return 2e-3
            else:
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
def gen_label():
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
def gen_dataset(n_batch):
    if dataset_name == 'celeba':
        # Spatial size of training images. All images will be resized to this size using a transformer.
        image_size_hr = (3, 128, 128)
    elif dataset_name == 'flickr':
        image_size_hr = (3, 128, 128)
    elif dataset_name == 'mnist':
        image_size_hr = (1, 28, 28)
    else:
        raise FileNotFoundError
    
    image_size_lr = image_size_hr[0], image_size_hr[1] // scale_factor, image_size_hr[2] // scale_factor
    
    if image_size_hr[1] % scale_factor or image_size_hr[2] % scale_factor:
        print("images trop petites", image_size_hr, image_size_lr)

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
    
    n = (len(dataset_hr) - batch_size)//2
    if not content_loss_on_lr:
        dataloader_hr = torch.utils.data.DataLoader(dataset_hr, sampler=utils.SamplerRange(0, 2*n), batch_size=batch_size, drop_last=True)
        size = len(dataloader_hr)
    else:
        class DoubleDataloader:
            def __init__(self, dataloader1, dataloader2):
                self.d1 = dataloader1
                self.d2 = dataloader2
            def __iter__(self):
                return zip(self.d1, self.d2)
        
        d1 = torch.utils.data.DataLoader(dataset_hr, sampler=utils.SamplerRange(0,   n), batch_size=batch_size, drop_last=True)
        d2 = torch.utils.data.DataLoader(dataset_hr, sampler=utils.SamplerRange(n, 2*n), batch_size=batch_size, drop_last=True)
        assert len(d1) == len(d2)
        size = len(d1)
        
        dataloader_hr = DoubleDataloader(d1, d2)
    
    test_hr = torch.cat([torch.unsqueeze(dataset_hr[i][0], 0) for i in range(-batch_size, 0)]).to(device)
    test_lr = utils.lr_from_hr(test_hr, image_size_lr[1:], device=device)
    
    if n_batch != -1:
        size = min(size, n_batch)
    
    return image_size_lr, image_size_hr, dataloader_hr, (test_hr, test_lr), size

# noinspection PyShadowingNames
def gen_device():
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = torch.cuda.device_count()
    
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    return device, ngpu

# noinspection PyShadowingNames
def gen_optimizers(checkpoint_path):
    optimizerG = optim.Adam(net_g.parameters(), lr=lr, betas=(.9, 0.999))
    optimizerD = optim.Adam(net_d.parameters(), lr=lr, betas=(.9, 0.999))
    
    if checkpoint_path is not None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            optimizerG.load_state_dict(checkpoint['opti_g'])
            optimizerD.load_state_dict(checkpoint['opti_d'])
        except Exception as e:
            print("erreur chargement optimizers:", e)
            pass
        
    return optimizerG, optimizerD

gen_seed()
device, _ngpu =                                                              gen_device()
image_size_lr, image_size_hr, dataloader_hr, (test_hr, test_lr), n_batch =   gen_dataset(n_batch)
net_g, net_d, identity, net_content_extractor, criterion, checkpoint_path =  gen_modules(_ngpu)
optimizerG, optimizerD =                                                     gen_optimizers(checkpoint_path)
loss_weight_adv_g, loss_weight_adv_d, loss_weight_cont =                     gen_losses(net_content_extractor, identity)
schedulerG, schedulerD =                                                     gen_scheduler(optimizerG, optimizerD)
real_label, real_label_reduced, fake_label =                                 gen_label()

