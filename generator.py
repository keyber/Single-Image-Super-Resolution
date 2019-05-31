import torch.nn as nn


class BasicBlock(nn.Module):
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

    @staticmethod
    def create_seq(n_blocks, n_features):
        return nn.Sequential(*[BasicBlock(n_features) for _ in range(n_blocks)])


class Generator(nn.Module):
    def __init__(self, n_blocks, n_features, output_channels=3):
        """input_shape: taille des images en entrée
        n_feature, n_blocks : ~expressivité du modèle
        output_channels: nombre de couleurs à sortir"""
        super().__init__()
        self.layers = nn.Sequential(
            # entrée
            nn.Conv2d(in_channels=3, out_channels=n_features, kernel_size=3, stride=1, padding=1), #9
            nn.PReLU(),

            # liste de blocks
            BasicBlock.create_seq(n_blocks, n_features),

            # fin des blocks
            nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=n_features),

            # upscale1
            # nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=3, stride=1, padding=1),
            # nn.PixelShuffle(upscale_factor=2),
            # nn.PReLU(),

            # upscale2
            # nn.Conv2d(in_channels=n_features // 4, out_channels=n_features // 4, kernel_size=3, stride=1, padding=1),
            # nn.PixelShuffle(upscale_factor=2),
            # nn.PReLU(),

            nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(),

            # sortie
            nn.Conv2d(in_channels=n_features // 4, out_channels=output_channels, kernel_size=3, stride=1, padding=1), #9
            # tanh ?
        )

    def forward(self, x):
        x = self.layers(x)
        return x
