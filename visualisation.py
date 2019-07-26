import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from model_generator import Generator, GeneratorSuffix
import utils
import torchvision.utils as vutils
import numpy as np
import torch.nn.functional
import matplotlib as mpl
import matplotlib.pyplot as plt

list_scales=[2]
progressive_gan_suffix = 1

factor = np.prod(list_scales) * (2 if progressive_gan_suffix else 1)
def main():
    net_g = Generator(n_blocks=16, n_features_block=64, n_features_last=256, list_scales=list_scales, input_channels=3, use_sn=True)
    if progressive_gan_suffix:
        net_g = GeneratorSuffix(net_g)
    
    net_g.eval()
    net_g.requires_grad = False
    
    path = input("entrer le chemin de sauvegarde du réseau à charger:\n")
    checkpoint = torch.load(path, map_location='cpu')
    net_g.load_state_dict(checkpoint['net_g'])
    
    dataset_name, dataroot = "celeba", "/local/beroukhim/data/celeba"
    # dataset_name, dataroot = "flickr", "/local/beroukhim/data/flickr/images"
    # dataset_name, dataroot = "HQ", "/local/beroukhim/data/flickr_HQ_faces"
    
    image_size_hr = 128
    image_size_lr = image_size_hr//factor
    image_size_uhr = image_size_hr*factor
    from PIL import Image
    dataset_hr = dset.ImageFolder(root=dataroot,
                                  transform=transforms.Compose([
                                      # transforms.CenterCrop(image_size_hr),
                                      transforms.Resize((image_size_hr, image_size_hr), interpolation=Image.BICUBIC),
                                      transforms.ToTensor(),
                                      transforms.Normalize((.5, .5, .5), (.5, .5, .5))]))
    n_im = 1
    # dataset_hr = [dataset_hr[i] for i in range(-10, 0)]
    dataloader_hr = torch.utils.data.DataLoader(dataset_hr, sampler=utils.SamplerRange(0, len(dataset_hr)), batch_size=n_im, num_workers=2)
    with torch.no_grad():
        for hr, _ in dataloader_hr:
            lr = utils.lr_from_hr(hr, (image_size_lr, image_size_lr))
            if torch.min(hr) < -1 or torch.max(hr)>1:
                print(torch.min(hr), torch.max(hr))
                print(torch.min(lr), torch.max(lr))
                print()
            sr = net_g(lr)
            ur = net_g(hr)
            list_img = [lr, sr, hr, ur]
            data = [np.transpose(vutils.make_grid(i, padding=0, normalize=True, nrow=1), (1, 2, 0)) for i in list_img]
            display_image(data, image_size_uhr, n_im)
            plt.show()

def display_image(data, image_size_uhr, n_im):
    n = len(data)
    dpi = mpl.rcParams['figure.dpi']
    figsize = image_size_uhr / dpi
    fig = plt.figure(figsize=(n * figsize, 2*figsize))
    width = 1/n
    titles_list = ['LR', 'SR', 'HR', 'UR']
    for i in range(len(data)):
        img = data[i]
        title = titles_list[i]
        ax = fig.add_axes([i*width, .5, width, .5])  # left, bottom, width, height
        ax.axis('off')
        plt.title(title)
        ax.imshow(img, interpolation='none')
        ax = fig.add_axes([i*width, .0, width, .5])
        ax.axis('off')
        ax.imshow(img, interpolation='bicubic')

main()
