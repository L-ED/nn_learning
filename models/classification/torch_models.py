import torch, torchvision
from torchvision.models import *

def resnet(name, version, classes, pretrained=True):
    
    model =  torch.hub.load(
        'pytorch/vision:v0.10.0', 
        name+version, 
        pretrained=True
    )
    if classes is not None:
        in_f = model.fc.in_features
        model.fc = torch.nn.Linear(in_f, classes)

    return model


def vgg(name, version, classes, pretrained=True):

    model =  torch.hub.load(
        'pytorch/vision:v0.10.0', 
        name+version, 
        pretrained=True
    )
    model.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
    if classes is not None:
        old_fc = getattr(model.classifier, "0")
        in_f = int(old_fc.in_features/49)
        new_fc = torch.nn.Linear(in_f, classes)
        delattr(model, "classifier")
        setattr(model, "classifier", new_fc) 
    return model


def mobilenet(name, version, classes, pretrained=True):

    model = torch.hub.load(
        'pytorch/vision:v0.10.0', 
        name + "_" + version,
        pretrained = True
    )
    if classes is not None:
        idx = "1" if "2" in version else "3"
        old_fc = getattr(model.classifier, idx)
        in_f = old_fc.in_features
        new_fc = torch.nn.Linear(in_f, classes)
        setattr(model.classifier, idx, new_fc)
    return model


MODELS={
    "vgg":
        '11':
        '13':

}