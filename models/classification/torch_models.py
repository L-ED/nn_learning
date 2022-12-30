import torch, torchvision
from torchvision.models import *


def transfer_learning(classifier_name):
    def replace(model, classes):
        classifier = getattr(model, classifier_name)

        if isinstance(classifier, torch.nn.Sequential):
            in_f = classifier[-1].in_features
            classifier[-1] = torch.nn.Linear(in_f, classes)

        elif isinstance(classifier, torch.nn.Linear):
            in_f = classifier.in_features
            classifier = torch.nn.Linear(in_f, classes)

        else:
            raise ValueError(
                "cant process model.{classifier_name} "+\
                f"{classifier.__class__}, "+\
                "Linear ad Sequential only suported"
            )

        setattr(model, classifier_name, classifier)
        return model

    return replace


def vgg(model, classes):

    model.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
    if classes is not None:
        old_fc = model.classifier[0]
        in_f = int(old_fc.in_features/49)
        model.classifier = torch.nn.Linear(in_f, classes) 

    return model


def squeeze(model, classes):
    in_c = model.classifier[1].in_channels
    model.classifier[1]=torch.nn.Conv2d(in_c, classes)
    return model


def swin(model, classes):
    in_c = model.head.in_features
    model.head=torch.nn.Linear(in_c, classes)
    return model


def vit(model, classes):

    in_c = model.heads.head.in_features
    model.heads.head = torch.nn.Linear(in_c, classes)

    return model


POSTPROCESSING_CLASSIFICATION={
    'vgg11': vgg, 
    'vgg11_bn': vgg, 
    'vgg13': vgg, 
    'vgg13_bn': vgg, 
    'vgg16': vgg, 
    'vgg16_bn': vgg, 
    'vgg19': vgg, 
    'vgg19_bn': vgg,

    'resnet101': transfer_learning("fc"), 
    'resnet152': transfer_learning("fc"), 
    'resnet18': transfer_learning("fc"), 
    'resnet34': transfer_learning("fc"), 
    'resnet50': transfer_learning("fc"),

    'wide_resnet101_2': transfer_learning("fc"), 
    'wide_resnet50_2': transfer_learning("fc"),

    'resnext101_32x8d': transfer_learning("fc"), 
    'resnext101_64x4d': transfer_learning("fc"), 
    'resnext50_32x4d': transfer_learning("fc"), 

    'mobilenet_v2': transfer_learning("classifier"), 
    'mobilenet_v3_large': transfer_learning("classifier"), 
    'mobilenet_v3_small': transfer_learning("classifier"),

    'convnext_base': transfer_learning("classifier"), # builder
    'convnext_large': transfer_learning("classifier"), # builder
    'convnext_small': transfer_learning("classifier"), # builder
    'convnext_tiny': transfer_learning("classifier"), # builder

    'densenet121': transfer_learning("classifier"), # builder
    'densenet161': transfer_learning("classifier"), # builder
    'densenet169': transfer_learning("classifier"), # builder
    'densenet201': transfer_learning("classifier"),

    'efficientnet_b0': transfer_learning("classifier"),  # builder
    'efficientnet_b1': transfer_learning("classifier"),  # builder
    'efficientnet_b2': transfer_learning("classifier"),  # builder
    'efficientnet_b3': transfer_learning("classifier"),  # builder
    'efficientnet_b4': transfer_learning("classifier"),  # builder
    'efficientnet_b5': transfer_learning("classifier"),  # builder
    'efficientnet_b6': transfer_learning("classifier"),  # builder
    'efficientnet_b7': transfer_learning("classifier"),  # builder
    'efficientnet_v2_l': transfer_learning("classifier"), # builder
    'efficientnet_v2_m': transfer_learning("classifier"), # builder
    'efficientnet_v2_s': transfer_learning("classifier"),

    'inception_v3': transfer_learning("fc"),
    'maxvit_t': transfer_learning("classifier"),
    'mnasnet0_5': transfer_learning("classifier"),  # builder
    'mnasnet0_75': transfer_learning("classifier"),  # builder
    'mnasnet1_0': transfer_learning("classifier"),  # builder
    'mnasnet1_3': transfer_learning("classifier"),

    'regnet_x_16gf': transfer_learning("fc"), 
    'regnet_x_1_6gf': transfer_learning("fc"), 
    'regnet_x_32gf': transfer_learning("fc"), 
    'regnet_x_3_2gf': transfer_learning("fc"), 
    'regnet_x_400mf': transfer_learning("fc"), 
    'regnet_x_800mf': transfer_learning("fc"), 
    'regnet_x_8gf': transfer_learning("fc"), 
    'regnet_y_128gf': transfer_learning("fc"), 
    'regnet_y_16gf': transfer_learning("fc"), 
    'regnet_y_1_6gf': transfer_learning("fc"), 
    'regnet_y_32gf': transfer_learning("fc"), 
    'regnet_y_3_2gf': transfer_learning("fc"), 
    'regnet_y_400mf': transfer_learning("fc"), 
    'regnet_y_800mf': transfer_learning("fc"), 
    'regnet_y_8gf': transfer_learning("fc"), 

    'shufflenet_v2_x0_5': transfer_learning("fc"), 
    'shufflenet_v2_x1_0': transfer_learning("fc"), 
    'shufflenet_v2_x1_5': transfer_learning("fc"), 
    'shufflenet_v2_x2_0': transfer_learning("fc"), 

    'squeezenet1_0': squeeze, 
    'squeezenet1_1': squeeze,

    'swin_b': swin, 
    'swin_s': swin, 
    'swin_t': swin, 
    'swin_v2_b': swin, 
    'swin_v2_s': swin, 
    'swin_v2_t': swin, 

    'vit_b_16': vit, 
    'vit_b_32': vit, 
    'vit_h_14': vit, 
    'vit_l_16': vit, 
    'vit_l_32': vit, 
 
}


# 'googlenet', # builder

# 'inception3', # base class  
# 'swintransformer',  # base class
# 'visiontransformer', # base class

# 'convnext_base', # builder
# 'convnext_large', # builder
# 'convnext_small', # builder
# 'convnext_tiny', # builder


# 'densenet121', # builder
# 'densenet161', # builder
# 'densenet169', # builder
# 'densenet201', # builder

# 'efficientnet_b0',  # builder
# 'efficientnet_b1',  # builder
# 'efficientnet_b2',  # builder
# 'efficientnet_b3',  # builder
# 'efficientnet_b4',  # builder
# 'efficientnet_b5',  # builder
# 'efficientnet_b6',  # builder
# 'efficientnet_b7',  # builder
# 'efficientnet_v2_l', # builder
# 'efficientnet_v2_m', # builder
# 'efficientnet_v2_s', # builder
 
 
# 'inception_v3', # builder

# 'maxvit_t', # builder

# 'mnasnet0_5',  # builder
# 'mnasnet0_75',  # builder
# 'mnasnet1_0',  # builder
# 'mnasnet1_3',  # builder

# 'mobilenet_v2', 
# 'mobilenet_v3_large', 
# 'mobilenet_v3_small', 

# 'regnet_x_16gf', 
# 'regnet_x_1_6gf', 
# 'regnet_x_32gf', 
# 'regnet_x_3_2gf', 
# 'regnet_x_400mf', 
# 'regnet_x_800mf', 
# 'regnet_x_8gf', 
# 'regnet_y_128gf', 
# 'regnet_y_16gf', 
# 'regnet_y_1_6gf', 
# 'regnet_y_32gf', 
# 'regnet_y_3_2gf', 
# 'regnet_y_400mf', 
# 'regnet_y_800mf', 
# 'regnet_y_8gf', 

# 'resnet101', 
# 'resnet152', 
# 'resnet18', 
# 'resnet34', 
# 'resnet50', 

# 'resnext101_32x8d', 
# 'resnext101_64x4d', 
# 'resnext50_32x4d', 

# 'shufflenet_v2_x0_5', 
# 'shufflenet_v2_x1_0', 
# 'shufflenet_v2_x1_5', 
# 'shufflenet_v2_x2_0', 

# 'squeezenet1_0', 
# 'squeezenet1_1', 

# 'swin_b', 
# 'swin_s', 
# 'swin_t', 

# 'swin_transformer', 
# 'swin_v2_b', 
# 'swin_v2_s', 
# 'swin_v2_t', 

# 'vgg11', 
# 'vgg11_bn', 
# 'vgg13', 
# 'vgg13_bn', 
# 'vgg16', 
# 'vgg16_bn', 
# 'vgg19', 
# 'vgg19_bn', 

# 'vision_transformer', 
# 'vit_b_16', 
# 'vit_b_32', 
# 'vit_h_14', 
# 'vit_l_16', 
# 'vit_l_32', 

# 'wide_resnet101_2', 
# 'wide_resnet50_2'