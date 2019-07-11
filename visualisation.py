import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from model_generator import Generator
import utils
import torchvision.utils as vutils
import numpy as np
import torch.nn.functional
import matplotlib as mpl
import matplotlib.pyplot as plt


def main():
    net_g = Generator(n_blocks=16, n_features_block=64, n_features_last=256, scale_twice=1, input_channels=3)
    net_g.eval()
    net_g.requires_grad = False
    
    path = input("entrer le chemin de sauvegarde du réseau à charger:\n")
    checkpoint = torch.load(path, map_location='cpu')
    net_g.load_state_dict(checkpoint['net_g'])
    
    dataset_name, dataroot = "celeba", "/local/beroukhim/data/celeba"
    # dataset_name, dataroot = "flickr", "/local/beroukhim/data/flickr/images"
    
    image_size_hr = 128
    image_size_lr = image_size_hr//4
    image_size_uhr = image_size_hr*4
    dataset_hr = dset.ImageFolder(root=dataroot,
                                  transform=transforms.Compose([
                                      # transforms.CenterCrop(image_size_hr),
                                      transforms.Resize((image_size_hr, image_size_hr)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((.5, .5, .5), (.5, .5, .5))]))
    n_im = 1
    dataloader_hr = torch.utils.data.DataLoader(dataset_hr, batch_size=n_im)
    
    with torch.no_grad():
        for hr, _ in dataloader_hr:
            lr = utils.lr_from_hr(hr, (image_size_lr, image_size_lr))
            sr = net_g(lr)
            usr = net_g(hr)
            list_img = [lr, sr, hr, usr]
            data = [np.transpose(vutils.make_grid(i, padding=0, normalize=True, nrow=1), (1, 2, 0)) for i in list_img]
            display_image(data, image_size_uhr, n_im)
            plt.show()

def display_image(data, image_size_uhr, n_im):
    n = len(data)
    dpi = mpl.rcParams['figure.dpi']
    figsize = image_size_uhr / dpi
    fig = plt.figure(figsize=(n * figsize, 2*figsize))
    width = 1/n
    for i, img in enumerate(data):
        ax = fig.add_axes([i*width, .5, width, .5])  # left, bottom, width, height
        ax.axis('off')
        ax.imshow(img, interpolation='none')
        ax = fig.add_axes([i*width, .0, width, .5])  # left, bottom, width, height
        ax.axis('off')
        ax.imshow(img, interpolation='bicubic')

main()
