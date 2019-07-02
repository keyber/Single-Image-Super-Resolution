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


class Generator(nn.Module):
    def __init__(self, n_blocks, n_features, forward_twice=False, scale_twice=False, input_channels=3):
        """n_blocks, n_features : ~expressivité du modèle
        input_channels: nombre de couleurs en entrée et en sortie
        scale_twice: False: x4 pixels, True: x16  pixels"""
        super().__init__()
        
        assert n_features % 4 == 0
        if scale_twice:
            assert n_features % 16 == 0
        
        self.forward_twice = forward_twice
        
        self.first_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=n_features, kernel_size=9, stride=1, padding=4),
            nn.PReLU())
        
        self.block_list = nn.Sequential(*[BasicBlock(n_features) for _ in range(n_blocks)])
        
        self.block_list_end = nn.Sequential(
            nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=n_features),
        )
        
        if not scale_twice:
            self.upscale = nn.Sequential(
                # upscale1
                nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
    
                # sortie
                nn.Conv2d(in_channels=n_features // 4, out_channels=input_channels, kernel_size=3, stride=1, padding=1),
                nn.Tanh())
        elif scale_twice==1:
            self.upscale = nn.Sequential(
                # upscale1
                nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
        
                # upscale2
                nn.Conv2d(in_channels=n_features // 4, out_channels=n_features // 4, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
        
                # sortie
                nn.Conv2d(in_channels=n_features // 16, out_channels=input_channels, kernel_size=3, stride=1, padding=1),
                nn.Tanh())
        elif scale_twice == 2:
            self.upscale = nn.Sequential(
                # upscale1
                nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(upscale_factor=4),
                nn.PReLU(),
    
                # sortie
                nn.Conv2d(in_channels=n_features // 16, out_channels=input_channels, kernel_size=3, stride=1, padding=1),
                nn.Tanh())
        elif scale_twice == 3:
            self.upscale = nn.Sequential(
                # upscale1
                nn.Conv2d(in_channels=n_features, out_channels=input_channels*16, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(upscale_factor=4),
                nn.Tanh())
            
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
        # print("gen", x.shape)
        return x
    
    def forward(self, x):
        x = self._sub_forward(x)
        
        if self.forward_twice:
            x = self._sub_forward(x)
            
        return x
