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
    def __init__(self, n_blocks, n_features_block, n_features_last, list_scales, use_sn=False, input_channels=3):
        """n_blocks, n_features : ~expressivité du modèle
        input_channels: nombre de couleurs en entrée et en sortie
        scale_twice: False: x4 pixels, True: x16  pixels"""
        super().__init__()
        
        assert n_features_last % 4 == 0
        self.n_features_last = n_features_last
        
        self.first_layers = nn.Sequential(
            sn(nn.Conv2d(in_channels=input_channels, out_channels=n_features_block, kernel_size=9, stride=1, padding=4)),
            nn.PReLU())
        
        self.block_list = nn.Sequential(*[BasicBlock(n_features_block) for _ in range(n_blocks)])
        
        self.block_list_end = nn.Sequential(
            sn(nn.Conv2d(in_channels=n_features_block, out_channels=n_features_block, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(num_features=n_features_block),
        )
        
        if use_sn:
            self.upscale = nn.Sequential(*[
                        nn.Sequential(sn(nn.Conv2d(in_channels=n_features_block if i==0 else n_features_last//list_scales[i-1]**2,
                                               out_channels=n_features_last, kernel_size=3, stride=1, padding=1)),
                                    nn.PixelShuffle(upscale_factor=list_scales[i]),
                                    nn.PReLU())
                    for i in range(len(list_scales))])
            self.end = nn.Sequential(
                    # sortie
                    sn(nn.Conv2d(in_channels=n_features_last//list_scales[-1]**2, out_channels=input_channels, kernel_size=3, stride=1, padding=1)),
                    nn.Tanh())
        else:
            self.upscale = nn.Sequential(*[
                        nn.Sequential(nn.Conv2d(in_channels=n_features_block if i==0 else n_features_last//list_scales[i-1]**2,
                                               out_channels=n_features_last, kernel_size=3, stride=1, padding=1),
                                    nn.PixelShuffle(upscale_factor=list_scales[i]),
                                    nn.PReLU())
                    for i in range(len(list_scales))])
            self.end = nn.Sequential(
                    nn.Conv2d(in_channels=n_features_last//list_scales[-1]**2, out_channels=input_channels, kernel_size=3, stride=1, padding=1),
                    nn.Tanh())
    
    def load_state_dict(self, state_dict, strict=False):
        super().load_state_dict(state_dict, strict=strict)
        
        a = self.state_dict()
        b = state_dict
        # noinspection PyTypeChecker
        if a.keys() != b.keys() or any(torch.any(a[k] != b[k]) for k in a.keys()): #différence de clé ou de valeur
            n_param_a = sum([x.nelement() for x in a.values()]) #somme des tailles des tensors
            n_param_b = sum([x.nelement() for x in b.values()])
            n_param_inter = sum([a[x].nelement() for x in set(a.keys()) & set(b.keys())])
            print("chargement du générateur à ", round(n_param_inter / n_param_a * 100, 1), "%",
                  "    (", round(n_param_inter*1e-6, 2), " M)", sep="")
            
            print("  - architecture : ", len(a), " ens de poids (", round(n_param_a*1e-6, 2), " M)", sep="")
            print("  - checkpoint   : ", len(b), " ens de poids (", round(n_param_b*1e-6, 2), " M)", sep="")
            
            manquants = a.keys() - b.keys()
            print("  - manquants    :", len(manquants), manquants)
            non_utilises = b.keys() - a.keys()
            print("  - non utilisés :", len(non_utilises), non_utilises)

    def forward_no_end(self, x):
        x = self.first_layers(x)
        residual = x
    
        x = self.block_list(x)
        x = self.block_list_end(x)
    
        x = x + residual
        x = self.upscale(x)
    
        return x

    def forward(self, x):
        x = self.forward_no_end(x)
        x = self.end(x)
        return x

    def freeze(self, freeze_upscale=False, freeze_end=False):
        layer_list = [self.first_layers, self.block_list, self.block_list_end]
        
        if freeze_upscale:
            layer_list.append(self.upscale)
            
        if freeze_end:
            layer_list.append(self.end)
        
        for layer in layer_list:
            layer.requires_grad=False
            for x in layer.parameters():
                x.requires_grad = False

class GeneratorSuffix(nn.Module):
    def __init__(self, prefix, freeze_prefix=False, **kwargs):
        super().__init__()
        self.base = prefix
        self.n_features_last = prefix.n_features_last
        self.upscale = nn.Sequential(*[
                        sn(nn.Conv2d(in_channels=self.n_features_last // 4, out_channels=self.n_features_last,
                                     kernel_size=3, stride=1, padding=1)),
                        nn.PixelShuffle(upscale_factor=2),
                        nn.PReLU()])
        # cache le parametre dans une liste pour qu'il ne soit vu qu'une seule fois
        self.end = [prefix.end[0] if type(prefix.end)==list else prefix.end]
        
        if freeze_prefix:
            prefix.freeze(**kwargs)
    
    def forward_no_end(self, x):
        x = self.base.forward_no_end(x)
        x = self.upscale(x)
        return x
        
    def forward(self, x):
        x = self.forward_no_end(x)
        x = self.end[0](x)
        return x

def _test_gen():
    from time import time
    
    for l in[[2], [2, 2], [2, 2, 2]]:
        g = Generator(16,64,256,l)
        g.load_state_dict(state_dict=torch.load("/local/beroukhim/srgan_trained/MSE_GANe-3_1epoch__1e-2_2epoch")['net_g'], strict=False)
        im = torch.empty([100,3,8,8])
        t = time()
        
        res = g(im)
        im2 = torch.empty(res.shape)
        loss = torch.sum(torch.pow(res - im2, 2))
        loss.backward()
        
        print(round(time() - t, 3), "s")
        assert res.shape == (100, 3, 8*2**len(l), 8*2**len(l)), res.shape

# noinspection PyTypeChecker
def _test_gen2():
    from time import time
    import torch.optim as optim
    import copy
    print("\nSUFFIX")
    g1 = Generator(16,64,256,[2])
    g2 = GeneratorSuffix(g1, freeze_prefix=True, freeze_upscale=True, freeze_end=True)
    p1 = copy.deepcopy(list(g1.parameters()))
    p2 = copy.deepcopy(list(g2.parameters()))
    im = torch.empty([100,3,8,8])
    adam = optim.Adam(g2.parameters(), lr=.1, betas=(.9, 0.999))
    t = time()
    
    res = g2(im)
    im2 = torch.empty(res.shape)
    loss = torch.sum(torch.pow(res - im2, 2))
    loss.backward()
    adam.step()
    
    print(round(time() - t, 3), "s")
    assert res.shape == (100, 3, 8*4, 8*4), res.shape
    assert not any(x is y for x,y in zip(p1, g1.parameters())) # deepcopy des params nécessaire
    assert all(torch.all(x==y) for x,y in zip(p1,g1.parameters())) # p1 inchangé
    assert any(torch.any(x!=y) for x,y in zip(p2,g2.parameters())) # p2 changé
    
if __name__ == '__main__':
    _test_gen()
    _test_gen2()
    print("tests passés")

#todo SpectralNorm ne marche pas avec load_state_dict(strict=False) https://github.com/pytorch/pytorch/pull/22545