import torch.nn as nn


class BasicBlock(nn.Module):
    """Block RESIDUEL utilisé par G"""
    def __init__(self, n_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=n_features),
            nn.PReLU(),
            nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=n_features))

    def forward(self, x):
        residual = x
        out = self.layers(x)
        return residual + out


class GeneratorProgresiveBase(nn.Module):
    def __init__(self, n_blocks, n_features, input_channels=3):
        """n_blocks, n_features : ~expressivité du modèle
        input_channels: nombre de couleurs en entrée et en sortie
        scale_twice: False: x4 pixels, True: x16  pixels"""
        super().__init__()
        
        
        self.first_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=n_features, kernel_size=9, stride=1, padding=4),
            nn.PReLU())
        
        self.block_list = nn.Sequential(*[BasicBlock(n_features) for _ in range(n_blocks)])
        
        self.block_list_end = nn.Sequential(
            nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=n_features),
        )
    
    def forward(self, x):
        x = self.first_layers(x)
        x = self.block_list(x)
        x = self.block_list_end(x)
        return x
    

class GeneratorSuffix(nn.Module):
    def __init__(self, prefix, n_features, input_channels=3):
        super().__init__()
        assert n_features % 4 == 0
        
        self.beginning = nn.Sequential(
            prefix,
            nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU())
        
        self.end = nn.Sequential(
            nn.Conv2d(in_channels=n_features // 4, out_channels=input_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh())
    
    def forward(self, x):
        x = self.beginning(x)
        x = self.end(x)
        return x

def _test():
    import torch

    x = torch.empty((1, 3, 32, 32))
    g0 = GeneratorProgresiveBase(16, n_features=64)
    g1 = GeneratorSuffix(g0, n_features=64)
    g2 = GeneratorSuffix(g1.beginning, n_features=16)
    g3 = GeneratorSuffix(g2.beginning, n_features=4)
    
    device = torch.device("cuda:0" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    print("device", device)
    if device=="cpu":
        y_list = [g(x) for g in [g1, g2, g3]]
    else:
        y_list = [g.to(device)(x.to(device)) for g in [g1, g2, g3]]
        
    for i, y in enumerate(y_list, 1):
        assert y.shape == (1, 3, 32*2**i, 32*2**i)
    
    print("tests passés")

if __name__ == '__main__':
    _test()