import torch, torchvision
from torchvision.models import *

def resnet(model, classes):
    if classes is not None:
        in_f = model.fc.in_features
        model.fc = torch.nn.Linear(in_f, classes)

    return model


def vgg(model, classes):

    model.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
    if classes is not None:
        old_fc = getattr(model.classifier, "0")
        in_f = int(old_fc.in_features/49)
        new_fc = torch.nn.Linear(in_f, classes)
        delattr(model, "classifier")
        setattr(model, "classifier", new_fc) 

    return model


def mobilenet(model, classes):

    if classes is not None:
        idx = "1" if "2" in version else "3"
        old_fc = getattr(model.classifier, idx)
        in_f = old_fc.in_features
        new_fc = torch.nn.Linear(in_f, classes)
        setattr(model.classifier, idx, new_fc)
    return model


POSTPROCESSING_CLASSIFICATION={
    "vgg":
        '13':

}


'googlenet', # builder

'inception3', # base class  
'swintransformer',  # base class of swin t
 
'visiontransformer', # base class

'convnext_base', # builder
'convnext_large', # builder
'convnext_small', # builder
'convnext_tiny', # builder


'densenet121', # builder
'densenet161', # builder
'densenet169', # builder
'densenet201', # builder

'efficientnet_b0',  # builder
'efficientnet_b1',  # builder
'efficientnet_b2',  # builder
'efficientnet_b3',  # builder
'efficientnet_b4',  # builder
'efficientnet_b5',  # builder
'efficientnet_b6',  # builder
'efficientnet_b7',  # builder
'efficientnet_v2_l', # builder
'efficientnet_v2_m', # builder
'efficientnet_v2_s', # builder
 
 
'inception_v3', # builder
'maxvit_t', # builder
'mnasnet0_5',  # builder
'mnasnet0_75',  # builder
'mnasnet1_0',  # builder
'mnasnet1_3',  # builder
'mobilenet', 
'mobilenet_v2', 
'mobilenet_v3_large', 
'mobilenet_v3_small', 

'regnet_x_16gf', 
'regnet_x_1_6gf', 
'regnet_x_32gf', 
'regnet_x_3_2gf', 
'regnet_x_400mf', 
'regnet_x_800mf', 
'regnet_x_8gf', 
'regnet_y_128gf', 
'regnet_y_16gf', 
'regnet_y_1_6gf', 
'regnet_y_32gf', 
'regnet_y_3_2gf', 
'regnet_y_400mf', 
'regnet_y_800mf', 
'regnet_y_8gf', 
'resnet101', 
'resnet152', 
'resnet18', 
'resnet34', 
'resnet50', 
'resnext101_32x8d', 
'resnext101_64x4d', 
'resnext50_32x4d', 
'shufflenet_v2_x0_5', 
'shufflenet_v2_x1_0', 
'shufflenet_v2_x1_5', 
'shufflenet_v2_x2_0', 
'squeezenet1_0', 
'squeezenet1_1', 
'swin_b', 
'swin_s', 
'swin_t', 
'swin_transformer', 
'swin_v2_b', 
'swin_v2_s', 
'swin_v2_t', 
'vgg11', 
'vgg11_bn', 
'vgg13', 
'vgg13_bn', 
'vgg16', 
'vgg16_bn', 
'vgg19', 
'vgg19_bn', 
'vision_transformer', 
'vit_b_16', 
'vit_b_32', 
'vit_h_14', 
'vit_l_16', 
'vit_l_32', 
'wide_resnet101_2', 
'wide_resnet50_2'