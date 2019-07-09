import torch.utils.data
import torch.nn
import torch.nn.functional

import numpy as np
from time import time
import utils
from config import *


def main():
    D_losses, G_losses, cont_loss, show_im = train_loop()

    # Affichage des résultats
    utils.save_and_show(net_g, net_d, optimizerG, optimizerD, D_losses, G_losses, cont_loss, show_im, write_root)


def train_loop():
    # ensemble d'anciennes images générées par G
    list_fakes = []
    
    # lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    cont_losses = []
    
    _zero = torch.zeros(1).to(device)
    print_period = max(1, n_batch//10)
    t = time()
    test_lr, test_hr = None, None
    
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, (img_hr, _) in enumerate(dataloader_hr):
            img_hr = img_hr.to(device)
            img_lr = utils.lr_from_hr(img_hr, image_size_lr[1:], device=device)
            
            if i == n_batch or i == len(dataloader_hr) - 1:
                test_lr, test_hr = img_lr.cpu(), img_hr.cpu()
                utils.save_curr_vis(img_list, img_lr, img_hr, net_g, G_losses, D_losses, cont_losses, plot_training)
                break
            
            real = img_hr
            
            # Generate fake image batch with G
            fake = net_g(img_lr)
            
            lw_adv_d = loss_weight_adv_d(epoch)
            if lw_adv_d:
                # Update the Discriminator with adversarial loss
                net_d.zero_grad()
                D_G_z1, D_x, errD = adversarial_loss_d(real, fake, list_fakes)
                errD *= lw_adv_d
                
                if normalized_gradient:
                    (errD / errD.item()).backward()
                else:
                    errD.backward()
                optimizerD.step()
            else:
                D_G_z1, D_x, errD  = 0, 0, _zero
            
            # sauvegarde un batch sur 10
            if i%10 == 0:
                old = fake.detach()
                # écrase un ancien aléatoirement pour ne pas prendre trop de RAM
                if len(list_fakes)==100:
                    list_fakes[random.randint(0,99)] = old
                else:
                    list_fakes.append(old)
            
            # Update the Generator
            net_g.zero_grad()

            # adversarial loss
            lw_adv_g = loss_weight_adv_g(epoch)
            if lw_adv_g:
                D_G_z2, errG_adv = adversarial_loss_g(fake)
                errG_adv *= lw_adv_g
            else:
                D_G_z2, errG_adv = 0, _zero
            
            # content loss
            lw_cont, content_extractor = loss_weight_cont(epoch)
            if lw_cont and content_extractor is not None :
                if content_loss_on_lr:
                    fake_bruitee = utils.lr_from_hr(fake, image_size_lr[1:], device=device)
                    err = content_loss_g(content_extractor, img_lr, fake_bruitee)
                else:
                    err = content_loss_g(content_extractor, real, fake)
                errG_cont = err * lw_cont
            else:
                errG_cont = _zero

            errG = errG_adv + errG_cont
            
            if lw_adv_g or lw_cont:
                if normalized_gradient:
                    (errG / errG.item()).backward()
                else:
                    errG.backward()
                optimizerG.step()
            
            # Output training stats
            if i % print_period == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G_adv: %.4f\tLoss_G_con: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader_hr),
                         errD.item(), errG_adv.item(), errG_cont.item(), D_x, D_G_z1, D_G_z2))
            
            # Save Losses for plotting later
            G_losses.append(errG_adv.item())
            D_losses.append(errD.item())
            cont_losses.append(errG_cont.item())

            schedulerD.step()
            schedulerG.step()
    
    print("train loop in", time() - t)
    return D_losses, G_losses, cont_losses, (test_lr, test_hr, img_list)


def adversarial_loss_d(real, curr_fake, old_fakes):
    """Update D network: maximize log(D(x)) + log(1 - D(G(z)))"""
    ### Train with all-real batch
    # Forward pass real batch through D
    d_real = net_d(real).view(-1)
    
    # Calculate loss on all-real batch
    errD_real = criterion(d_real, real_label_reduced)
    
    # Calculate gradients for D in backward pass
    # errD_real.backward()
    D_x = d_real.mean().item()
    
    errD = errD_real
    D_G_z1 = torch.zeros(1)
    
    list_fakes = [curr_fake]
    list_fakes += list(np.random.choice(old_fakes, len(old_fakes)//10, replace=False))
    
    for fake in list_fakes:
        ## Train with all-fake batch
        # Classify all fake batch with D
        d_fake = net_d(fake.detach()).view(-1)
        
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(d_fake, fake_label)
        
        # Calculate the gradients for this batch
        # errD_fake.backward()
        D_G_z1 += d_fake.mean().item()
        
        # Add the gradients from the all-real and all-fake batches
        errD += errD_fake
    
    return D_G_z1, D_x, errD


def adversarial_loss_g(fake):
    """Update G network: maximize log(D(G(z)))"""
    # Since we just updated D, perform another forward pass of all-fake batch through D
    output = net_d(fake).view(-1)
    
    # Calculate G's loss based on this output
    errG = criterion(output, real_label)
    
    # Calculate gradients for G
    D_G_z2 = output.mean().item()
    return D_G_z2, errG

def content_loss_g(content_extractor, real, fake):
    a = content_extractor(real)
    b = content_extractor(fake)
    return torch.mean(torch.pow(a - b, 2))

if __name__ == '__main__':
    main()
