import torchvision.models as models


def vgg():
    """les convolutions peuvent être appliquées quelle que soit la taille de l'image"""
    vgg_ = models.vgg19(pretrained=True)
    vgg_ = vgg_.features
    vgg_.eval()
    vgg.requires_grad = False
    for param in vgg_.parameters():
        assert param.requires_grad
        param.requires_grad = False
    print(vgg_)
    return vgg_

def vgg_2conv_1maxPool():
    vgg_ = models.vgg19(pretrained=True)
    vgg_ = vgg_.features[:5]
    vgg_.eval()
    vgg.requires_grad = False
    for param in vgg_.parameters():
        assert param.requires_grad
        param.requires_grad = False
    print(vgg_)
    return vgg_

def vgg_4conv_2maxPool():
    vgg_ = models.vgg19(pretrained=True)
    vgg_ = vgg_.features[:10]
    vgg_.eval()
    vgg.requires_grad = False
    for param in vgg_.parameters():
        assert param.requires_grad
        param.requires_grad = False
    print(vgg_)
    return vgg_

vgg_2conv_1maxPool()
vgg_4conv_2maxPool()
exit()