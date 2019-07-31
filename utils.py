import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.nn.functional
import os
import multiprocessing
import gc
import pickle


print_process = None


def _subsampling_interpolation(img_hr, image_size_lr):
    return torch.nn.functional.interpolate(img_hr, image_size_lr, mode='bicubic', align_corners=True)

def _crop_lr(img_lr, device='cpu'):
    return torch.max(torch.min(img_lr, torch.full((1,), 1.0, device=device)), torch.full((1,), -1.0, device=device))
    
def lr_from_hr(img_hr, image_size_lr, device='cpu'):
    """
    hr dans [-1, 1]
    lr = interpolation(hr) dépasse de [-1, 1]
    on seuille les valeurs qui dépassent
    (le lr de SRGAN est dans [0, 1])
    """
    img_lr = _subsampling_interpolation(img_hr, image_size_lr)
    img_lr = _crop_lr(img_lr, device)
    return img_lr

def _test_lr_from_hr():
    # l'interpolation fait sortir de [-1, 1]
    max_val = 0
    for i in range(1000):
        im_lr = _subsampling_interpolation(torch.rand((1,1,8,8)) * 2 - 1, (4,4))
        max_val = max(max_val, torch.max(torch.abs(im_lr)))
    assert max_val > 1.1
    
    # crop quand on est dans les bornes est inutile
    im_lr0 = torch.tensor([[[[1., -1.], [-1., 1.]]]])
    assert torch.all(_crop_lr(im_lr0) == im_lr0)
    
    # test du crop
    im_lr1 = torch.tensor([[[[1.9, -1.], [-1., 1.]]]])
    assert torch.all(_crop_lr(im_lr1) == im_lr0)
    
    
def save_curr_vis(img_list, img_lr, img_hr, netG, G_losses, D_losses, cont_losses, plot_training):
    global print_process

    with torch.no_grad():
        fake_sr = netG(img_lr[:4]).detach().cpu()
        if img_hr is not None:
            fake_usr = netG(img_hr[:4]).detach().cpu()
    
    if img_hr is not None:
        img_list.append((vutils.make_grid(fake_sr, padding=0, normalize=True, nrow=2),
                         vutils.make_grid(fake_usr, padding=0, normalize=True, nrow=2)))
    else:
        img_list.append((vutils.make_grid(fake_sr, padding=0, normalize=True, nrow=2),))
    
    def f():
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(np.transpose(img_list[-1][0], (1, 2, 0)))
        plt.subplot(1, 2, 2)
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.plot(cont_losses, label="cont")
        plt.legend()
        plt.show()
    
    if print_process is not None:
        print_process.terminate()
    
    if plot_training:
        print_process = multiprocessing.Process(target=f)
        print_process.start()


def save_and_show(epoch, net_g, net_d, optimizerG, optimizerD, D_losses, G_losses, cont_losses, show_im, dis_list_old, write_root):
    if print_process is not None:
        print_process.terminate()
    
    # sauvegarde le réseau
    write_path_ = _save(epoch, net_g, net_d, optimizerG, optimizerD, dis_list_old, write_root)
    
    # attend que l'utilisateur soit là pour créer des figures
    input("appuyer sur une touche pour afficher")
    
    _plot(D_losses, G_losses, cont_losses, show_im, write_path_)
    _anim(show_im, write_path_)


def _save(epoch, net_g, net_d, optimizerG, optimizerD, dis_list_old, write_root):
    if not input("sauvegarder ? Y/n") == "n":
        if not os.path.isdir(write_root):
            os.mkdir(write_root)
        
        i = 0
        write_path = write_root + str(i)
        while os.path.isfile(write_path) or os.path.isfile(write_path + "_ani.mp4"):
            i += 1
            write_path = write_root + str(i)
        
        torch.save({
            'epoch': epoch,
            'net_g': net_g.state_dict(),
            'net_d': net_d.state_dict(),
            'opti_g': optimizerG.state_dict(),
            'opti_d': optimizerD.state_dict(),
            'dis_list': dis_list_old
        }, write_path)
        
        print("réseau sauvegardé dans le fichier", write_path)
        return write_path
    return None


def _plot(D_losses, G_losses, cont_losses, show_im, write_path):
    test_lr, test_hr, img_list = show_im
    
    try:
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.plot(cont_losses, label="cont")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        
        plt.figure(figsize=(8, 8))
        # Plot the LR images
        plt.subplot(2, 2, 1)
        plt.axis("off")
        plt.title("LR Images")
        plt.imshow(np.transpose(
            vutils.make_grid(test_lr[:4].detach().cpu(), padding=0, normalize=True, nrow=2), (1, 2, 0)))
        
        # Plot the HR images
        plt.subplot(2, 2, 3)
        plt.axis("off")
        plt.title("HR Images")
        plt.imshow(np.transpose(
            vutils.make_grid(test_hr[:4].detach().cpu(), padding=0, normalize=True, nrow=2).cpu(), (1, 2, 0)))
        
        # Plot the SR from the last epoch
        plt.subplot(2, 2, 2)
        plt.axis("off")
        plt.title("SR Images")
        plt.imshow(np.transpose(img_list[-1][0], (1, 2, 0)))
        
        if len(img_list[-1]) == 2:
            # Plot the SR from the last epoch
            plt.subplot(2, 2, 4)
            plt.axis("off")
            plt.title("USR Images")
            plt.imshow(np.transpose(img_list[-1][1], (1, 2, 0)))
        
        plt.show()

    except Exception as e:
        print("affichage loss échoué", e)
        if write_path is not None:
            with open(write_path + ".loss", "wb") as f:
                pickle.dump({'G': G_losses,
                         'D': D_losses,
                         'cont':cont_losses}, f)


def _anim(show_im, write_path):
    _, _, img_list = show_im

    try:
        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        # np.transpose inverse les axes pour remettre le channel des couleurs en dernier
        ims = [[plt.imshow(np.transpose(i[0], (1, 2, 0)), animated=True)] for i in img_list]
        
        # il faut stocker l'animation dans une variable sinon l'animation plante
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        
        writer = animation.writers['ffmpeg'](fps=10, bitrate=1800)
        
        if write_path is not None:
            ani.save(write_path + ".mp4", writer=writer)
        
        plt.show()
    except Exception as e:
        print("affichage animation échoué", e)
        if write_path is not None:
            with open(write_path + ".list", "wb") as f:
                pickle.dump(img_list, f)


class SamplerRange(torch.utils.data.sampler.Sampler):
    def __init__(self, a, b):
        super().__init__(a)
        self.a = a
        self.b = b
    
    def __iter__(self):
        return iter(range(self.a, self.b))
    
    def __len__(self):
        return self.b - self.a


def mem_report():
    s = 0
    gc.collect()
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            s += obj.nelement()
    if s>mem_report.max_size:
        mem_report.max_size=s
        print("%.1e" % (s*4*2**-30))
mem_report.max_size = 0

if __name__ == '__main__':
    _test_lr_from_hr()
    print("tests passés")