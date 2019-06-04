import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, n_features, stride):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=n_features, out_channels=n_features, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(num_features=n_features),
            nn.LeakyReLU())
    
    def forward(self, x):
        return self.layers(x)

    @staticmethod
    def create_seq(list_n_features, list_stride):
        return nn.Sequential(*[BasicBlock(n_features, stride) for
                               (n_features, stride) in zip(list_n_features, list_stride)])


class Discriminator(nn.Module):
    def __init__(self, input_shape, list_n_features, list_stride):
        """
        [64, 128, 128, 256, 256, 512, 512],
        [2 ,   1,   2,   1,   2,   1,   2]"""
        super().__init__()
        w = input_shape[1]
        h = input_shape[2]
        assert w==h
        for x in list_stride:
            assert x in (1,2)
        assert w*h % 4**(sum(list_stride) - len(list_stride)) == 0
        
        self.fc_in = w*h*list_n_features[-1]//(4**(sum(list_stride) - len(list_stride)))

        self.conv = nn.Sequential(
            # entrée
            nn.Conv2d(in_channels=input_shape[0], out_channels=list_n_features[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=list_n_features[0], out_channels=list_n_features[1], kernel_size=3, stride=list_stride[0], padding=1),
            nn.BatchNorm2d(num_features=list_n_features[1]),
            nn.LeakyReLU(),

            # liste de blocks
            BasicBlock.create_seq(list_n_features[1:], list_stride[1:])
        )

        self.fc = nn.Sequential(
            # sortie
            nn.Linear(self.fc_in, list_n_features[-1]*2),
            nn.LeakyReLU(),

            nn.Linear(list_n_features[-1]*2, 1),
            nn.Sigmoid())

    def forward(self, x):
        # print("dis", x.shape)
        x = self.conv(x)
        x = x.view(-1, self.fc_in)
        x = self.fc(x)
        # print("dis", x.shape)
        return x

