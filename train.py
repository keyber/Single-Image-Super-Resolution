import torch.utils.data
import torch.nn
import torch.nn.functional

from time import time
import utils
from config import *


def main():
    # Initialize BCELoss function  #binary cross entropy
    criterion = torch.nn.BCELoss()
    
    dataloader_hr = torch.utils.data.DataLoader(dataset_hr, batch_size=batch_size, num_workers=2)
    
    D_losses, G_losses, cont_loss, show_im = train_loop(criterion, dataloader_hr)

    # Affichage des résultats
    utils.save_and_show(D_losses, G_losses, cont_loss, show_im)


def train_loop(criterion, dataloader_hr):
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    cont_losses = []
    i_tot = 0
    
    _zero = torch.zeros(1).to(device)
    print_period = max(1, (n_batch if n_batch!=-1 else len(dataloader_hr))//10)
    t = time()
    test_lr, test_hr = None, None
    
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, (img_hr, _) in enumerate(dataloader_hr):
            img_hr = img_hr.to(device)
            img_lr = torch.nn.functional.interpolate(img_hr, scale_factor=1/4 if scale_twice else 1/2,mode='bicubic', align_corners=False)
            
            if i == n_batch or i == len(dataloader_hr)-1:
                test_lr, test_hr = img_lr, img_hr
                utils.save_curr_vis(img_list, test_lr, test_hr, net_g, G_losses, D_losses, cont_losses)
                break
            
            real = img_hr
            
            # Generate fake image batch with G
            fake = net_g(img_lr)
            
            lw_adv_d = loss_weight_adv_d(i_tot)
            if lw_adv_d:
                # Update the Discriminator with adversarial loss
                net_d.zero_grad()
                D_G_z1, D_x, errD = lw_adv_d * adversarial_loss_d(criterion, real, fake)
                optimizerD.step()
            else:
                D_G_z1, D_x, errD  = 0, 0, _zero
            
            # Update the Generator
            net_g.zero_grad()

            # adversarial loss
            lw_adv_g = loss_weight_adv_g(i_tot)
            if lw_adv_g:
                D_G_z2, errG_adv = lw_adv_g * adversarial_loss_g(criterion, fake)
            else:
                D_G_z2, errG_adv = 0, _zero
            
            # content loss
            lw_cont, content_extractor = loss_weight_cont(i_tot)
            if lw_cont and content_extractor is not None :
                if content_loss_on_lr:
                    fake_bruitee = torch.nn.functional.interpolate(fake, scale_factor=1/4 if scale_twice else 1/2,mode='bicubic', align_corners=False)
                    err = content_loss_g(content_extractor, img_lr, fake_bruitee)
                else:
                    err = content_loss_g(content_extractor, real, fake)
                errG_cont = lw_cont * err
            else:
                errG_cont = _zero
            
            errG = errG_adv + errG_cont
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
            i_tot += 1

        schedulerD.step()
        schedulerG.step()
    
    print("train loop in", time() - t)
    return D_losses, G_losses, cont_losses, (test_lr, test_hr, img_list)


def adversarial_loss_d(criterion, real, fake):
    """Update D network: maximize log(D(x)) + log(1 - D(G(z)))"""
    ### Train with all-real batch
    # Forward pass real batch through D
    d_real = net_d(real).view(-1)
    
    # Calculate loss on all-real batch
    label = torch.full((fake.size(0),), real_label, device=device)
    errD_real = criterion(d_real, label)
    
    # Calculate gradients for D in backward pass
    errD_real.backward()
    D_x = d_real.mean().item()
    
    ## Train with all-fake batch
    # Classify all fake batch with D
    d_fake = net_d(fake.detach()).view(-1)
    
    # Calculate D's loss on the all-fake batch
    label.fill_(fake_label)
    errD_fake = criterion(d_fake, label)
    
    # Calculate the gradients for this batch
    errD_fake.backward()
    D_G_z1 = d_fake.mean().item()
    
    # Add the gradients from the all-real and all-fake batches
    errD = errD_real + errD_fake
    return D_G_z1, D_x, errD


def adversarial_loss_g(criterion, fake):
    """Update G network: maximize log(D(G(z)))"""
    # Since we just updated D, perform another forward pass of all-fake batch through D
    output = net_d(fake).view(-1)
    
    # Calculate G's loss based on this output
    label = torch.full((fake.size(0),), real_label, device=device) # fake labels are real for generator cost
    errG = criterion(output, label)
    
    # Calculate gradients for G
    D_G_z2 = output.mean().item()
    return D_G_z2, errG

def content_loss_g(content_extractor, real, fake):
    a = content_extractor(real)
    b = content_extractor(fake)
    return torch.mean(torch.pow(a - b, 2))


if __name__ == '__main__':
    main()
