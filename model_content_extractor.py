import torchvision.models as models
import torch.nn as nn
import torch

# indices des couches maxPool dans VGG19 (on n'utilise jamais la dernière)
maxPoolInd = ( 4,   9,  18,  27,  36)

# taille des feature map (pour les tests)
layersSize = (64, 128, 256, 512, 512) # les deux derniers sont bien à 512

def identity():
    """mène à une simple MSE entre les deux images"""
    return nn.Identity()
    
def vgg_4conv_1maxPool():
    """récupère la feature map de VGG avant le deuxième maxPool
    shape: (b_s, 128, 64, 64)"""
    vgg_ = models.vgg19(pretrained=True)
    
    # on garde toutes les couches avant la deuxième convolution
    vgg_ = vgg_.features[:9]
    
    # on désactive tout ce qu'on peut pour être sûr de ne pas backpropager dans VGG
    vgg_.eval()
    vgg_.requires_grad = False
    for param in vgg_.parameters():
        assert param.requires_grad
        param.requires_grad = False
    
    return vgg_

class MaskedVGG(nn.Module):
    """concatène les feature map de VGG avant les maxPool dont le bit correspondant vaut 1
    shape: (b_s, -1)"""
    def __init__(self, mask):
        super().__init__()

        # on garde toutes les couches demandées
        self.intermediate_layers_kept = [maxPoolInd[i] for i in range(len(maxPoolInd)) if mask & (1<<i)]
        
        self.layers = models.vgg19(pretrained=True).features[:self.intermediate_layers_kept[-1]]
        self.layers.eval()
        self.layers.requires_grad = False
        for param in self.layers.parameters():
            assert param.requires_grad
            param.requires_grad = False
        
        
    def forward(self, x):
        saved = []
        
        for i, l in enumerate(self.layers, 1):
            x = l(x)

            if i in self.intermediate_layers_kept:
                saved.append(x)
        
        return torch.cat([e.view(e.shape[0], -1) for e in saved], dim=1)


def get_size(im, mask):
    assert im.shape[1]==3
    
    w, h = im.shape[2], im.shape[3]
    
    size = 0
    for i in range(len(layersSize)):
        if mask & (1<<i):
            size += (w//2**i) * (h//2**i) * layersSize[i]
            
    return size

def _test_base():
    m = MaskedVGG(0b00011)
    assert len(m.layers)==9
    m = MaskedVGG(0b00010)
    assert len(m.layers)==9
    m = MaskedVGG(0b00110)
    assert len(m.layers)==18
    
    # les maxpool divisent w et h par 2
    im = torch.empty((1, 3, 32, 32))
    out = nn.MaxPool2d(kernel_size=2, stride=2)(im)
    assert out.shape == (1, 3, 16, 16)
    
    # les dimensions sont tronquées
    im = torch.empty((1, 3, 33, 33))
    out = nn.MaxPool2d(kernel_size=2, stride=2)(im)
    assert out.shape == (1, 3, 16, 16)
    print("test base passés")

def _test_mask():
    for mask in range(1, 2**5):
        net = MaskedVGG(mask)
        im = torch.empty((1, 3, 64, 64))
        features = net(im)
    
        assert features.shape == (1, get_size(im, mask))
        print("mask {0:7b} \ttaille {1:d}".format(mask, features.shape[1]))
        # print("mask: %d\ttaille:%d" %(mask, features.shape[1]))
    print("test masked_vgg passés")

if __name__ == '__main__':
    _test_base()
    _test_mask()
