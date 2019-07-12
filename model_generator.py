import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as sn

class BasicBlock(nn.Module):
    """Block RESIDUEL utilisé par G"""
    def __init__(self, n_features):
        super().__init__()
        self.layers = nn.Sequential(
            sn(nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(num_features=n_features),
            nn.PReLU(),
            sn(nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(num_features=n_features))

    def forward(self, x):
        residual = x
        out = self.layers(x)
        return residual + out


class Generator(nn.Module):
    def __init__(self, n_blocks, n_features_block, n_features_last, list_scales, forward_twice=False, input_channels=3):
        """n_blocks, n_features : ~expressivité du modèle
        input_channels: nombre de couleurs en entrée et en sortie
        scale_twice: False: x4 pixels, True: x16  pixels"""
        super().__init__()
        
        assert n_features_last % 4 == 0
        
        self.forward_twice = forward_twice
        
        self.first_layers = nn.Sequential(
            sn(nn.Conv2d(in_channels=input_channels, out_channels=n_features_block, kernel_size=9, stride=1, padding=4)),
            nn.PReLU())
        
        self.block_list = nn.Sequential(*[BasicBlock(n_features_block) for _ in range(n_blocks)])
        
        self.block_list_end = nn.Sequential(
            sn(nn.Conv2d(in_channels=n_features_block, out_channels=n_features_block, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(num_features=n_features_block),
        )
        
        self.upscale = nn.Sequential(*[
                    nn.Sequential(nn.Conv2d(in_channels=n_features_block if i==0 else n_features_last//list_scales[i-1]**2,
                                           out_channels=n_features_last, kernel_size=3, stride=1, padding=1),
                                nn.PixelShuffle(upscale_factor=list_scales[i]),
                                nn.PReLU())
                for i in range(len(list_scales))])

        self.end = nn.Sequential(
                # sortie
                nn.Conv2d(in_channels=n_features_last//list_scales[-1]**2, out_channels=input_channels, kernel_size=3, stride=1, padding=1),
                nn.Tanh())
    
    def load_state_dict(self, state_dict, strict=False):
        super().load_state_dict(state_dict, strict=strict)
        
        a = self.state_dict()
        b = state_dict
        if a != b:
            n_param_a = sum([x.nelement() for x in set(a.values())])
            n_param_b = sum([x.nelement() for x in set(b.values())])
            n_param_inter = sum([a[x].nelement() for x in set(a.keys()) & set(b.keys())])
            print("chargement du générateur à ", round(n_param_inter / n_param_a * 100, 1), "%",
                  "    (", round(n_param_inter*1e-6, 2), " M)", sep="")
            
            print("  - architecture : ", len(a), " ens de poids (", round(n_param_a*1e-6, 2), " M)", sep="")
            print("  - checkpoint   : ", len(b), " ens de poids (", round(n_param_b*1e-6, 2), " M)", sep="")
            
            manquants = a.keys() - b.keys()
            print("  - manquants    :", len(manquants), manquants)
            non_utilises = b.keys() - a.keys()
            print("  - non utilisés :", len(non_utilises), non_utilises)
            
            
    def _sub_forward(self, x):
        # print("gen", x.shape)
        # for l in self.layers:
        #     print(l)
        #     x = l(x)
        #     print(x.shape)
        
        x = self.first_layers(x)
        residual = x
        
        x = self.block_list(x)
        x = self.block_list_end(x)
        
        x = x + residual
        x = self.upscale(x)
        x = self.end(x)
        
        return x
    
    def forward(self, x):
        x = self._sub_forward(x)
        
        if self.forward_twice:
            x = self._sub_forward(x)
        
        return x

def _test():
    from time import time
    import torch.optim as optim
    
    for l in[[2], [2, 2], [2, 2, 2]]:
        g = Generator(16,64,256,l)
        g.load_state_dict(state_dict=torch.load("/local/beroukhim/srgan_trained/MSE_GANe-3_1epoch__1e-2_2epoch")['net_g'], strict=False)
        im = torch.empty([100,3,8,8])
        t = time()
        
        adam = optim.Adam(g.parameters(), lr=.001, betas=(.9, 0.999))
        res = g(im)
        im2 = torch.empty(res.shape)
        loss = torch.sum(torch.pow(res - im2, 2))
        loss.backward()
        adam.step()
        
        print(round(time() - t, 3), "s")
        assert res.shape == (100, 3, 8*2**len(l), 8*2**len(l)), res.shape
        
if __name__ == '__main__':
    _test()

#todo SpectralNorm ne marche pas avec load_state_dict(strict=False) https://github.com/pytorch/pytorch/pull/22545