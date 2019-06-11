import torch.nn as nn


class BasicBlock(nn.Module):
    """Block NON RESIDUEL utilisé par D"""
    def __init__(self, n_in, n_out, stride):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(num_features=n_out),
            nn.LeakyReLU())
    
    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self, input_shape, list_n_features, list_stride):
        """
        
        features et strides utilisés dans SRGAN :
            [64, 64, 128, 128, 256, 256, 512, 512],
            [1,   2,   1,   2,   1,   2,   1,   2]"""
        super().__init__()
        w = input_shape[1]
        h = input_shape[2]
        for x in list_stride:
            assert x in (1, 2), "l'article utilise des stride de 1 ou 2 seulement"
        assert w * h % 4 ** (sum(list_stride) - len(list_stride)) == 0,\
            "chaque stride à 2 divisise la taille par 2, il faut que ca soit divisible"
        
        assert len(list_n_features) == len(list_stride)
        
        # taille des vecteurs d'entrée de la couche FC (calcul manuel pour être sûr, mais on pourrait mettre -1)
        self.fc_in = w * h * list_n_features[-1] // (4 ** (sum(list_stride) - len(list_stride)))
        
        self.conv = nn.Sequential(
            # entrée
            nn.Conv2d(in_channels=input_shape[0], out_channels=list_n_features[0], kernel_size=3, stride=list_stride[0], padding=1),
            nn.LeakyReLU(),
            
            # liste de blocks
            nn.Sequential(*[BasicBlock(list_n_features[i - 1], list_n_features[i], list_stride[i])
                            for i in range(1, len(list_n_features))])
        )
        
        self.fc = nn.Sequential(
            # sortie
            nn.Linear(self.fc_in, list_n_features[-1] * 2),
            nn.LeakyReLU(),
            
            nn.Linear(list_n_features[-1] * 2, 1),
            nn.Sigmoid())
    
    def forward(self, x):
        # print("dis", x.shape)
        x0 = x
        x = self.conv(x)
        x = x.view(x0.shape[0], self.fc_in)
        x = self.fc(x)
        # print("dis", x.shape)
        return x
