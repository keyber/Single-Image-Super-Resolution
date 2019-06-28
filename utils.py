import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.nn.functional
import os
from config import *
import multiprocessing


print_process = None


def save_curr_vis(img_list, img_lr, img_hr, netG, G_losses, D_losses, cont_losses):
    global print_process
    
    with torch.no_grad():
        fake_sr = netG(img_lr[:4]).detach().cpu()
        fake_usr = netG(img_hr[:4]).detach().cpu()
    img_list.append((vutils.make_grid(fake_sr, padding=0, normalize=True, nrow=2),
                     vutils.make_grid(fake_usr, padding=0, normalize=True, nrow=2)))
    
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


def save_and_show(D_losses, G_losses, cont_losses, show_im):
    if print_process is not None:
        print_process.terminate()
    
    # sauvegarde le réseau
    write_path_ = _save()
    
    # attend que l'utilisateur soit là pour créer des figures
    input("appuyer sur une touche pour afficher")
    
    _plot(D_losses, G_losses, cont_losses, show_im)
    _anim(show_im, write_path_)


def _save():
    if not input("sauvegarder ? Y/n") == "n":
        if not os.path.isdir(write_root):
            os.mkdir(write_root)
        
        i = 0
        write_path = write_root + str(i)
        while os.path.isfile(write_path) or os.path.isfile(write_path + "_ani.mp4"):
            i += 1
            write_path = write_root + str(i)
        
        torch.save({
            'net_g': net_g.state_dict(),
            'net_d': net_d.state_dict(),
            'opti_g': optimizerG.state_dict(),
            'opti_d': optimizerD.state_dict()
        }, write_path)
        
        print("réseau sauvegardé dans le fichier", write_path)
        return write_path
    return None


def _plot(D_losses, G_losses, cont_losses, show_im):
    test_lr, test_hr, img_list = show_im
    
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.plot(cont_losses, label="cont")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.figure(figsize=(15, 8))
    # Plot the LR images
    plt.subplot(2, 2, 1)
    plt.axis("off")
    plt.title("LR Images")
    plt.imshow(np.transpose(
        vutils.make_grid(test_lr.to(device)[:4], padding=0, normalize=True, nrow=2).cpu(), (1, 2, 0)))
    
    # Plot the HR images
    plt.subplot(2, 2, 3)
    plt.axis("off")
    plt.title("HR Images")
    plt.imshow(np.transpose(
        vutils.make_grid(test_hr.to(device)[:4], padding=0, normalize=True, nrow=2).cpu(), (1, 2, 0)))
    
    # Plot the SR from the last epoch
    plt.subplot(2, 2, 2)
    plt.axis("off")
    plt.title("SR Images")
    plt.imshow(np.transpose(img_list[-1][0], (1, 2, 0)))
    
    # Plot the SR from the last epoch
    plt.subplot(2, 2, 4)
    plt.axis("off")
    plt.title("USR Images")
    plt.imshow(np.transpose(img_list[-1][1], (1, 2, 0)))
    plt.show()


def _anim(show_im, write_path):
    _, _, img_list = show_im
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    # np.transpose inverse les axes pour remettre le channel des couleurs en dernier
    ims = [[plt.imshow(np.transpose(i[0], (1, 2, 0)), animated=True)] for i in img_list]
    
    # il faut stocker l'animation dans une variable sinon l'animation plante
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    
    if write_path is None:
        return
    writer = animation.writers['ffmpeg'](fps=1, bitrate=1800)
    ani.save(write_path + "_ani.mp4", writer=writer)
    
    plt.show()
