import torch, torchvision, os, sys, json
from ..datasets import generic_set_one_annotation
import yaml
from platform_converter.models.models.networks.classification.ofa_mobilenetv3 import MobileNetVPlatComp
from platform_converter.models.models.networks.classification.ofa_resnet import ResNets #ResNetsQ


def create_dict_from_module(module):
    return {
        name.lower(): getattr(module, name) 
        for name in dir(module) 
        if not name.startswith("__")
    }

def create_module_from_source(source):
    parent_path = os.path.dirname(source)
    dir_name = os.path.basename(source).split('.')[0]
    sys.path.insert(0, parent_path)
    print(parent_path)
    return __import__(dir_name)


def get_(dict_, name, version=None):
    try:
        ret_=dict_[name]
        if version is not None:
            try:
                return ret_[version]
            except KeyError as e:
                supported_versions = ', '.join([f'{i}' for i in ret_.keys()])
                raise KeyError(f'{name} has no version {version}, only {supported_versions} supported')
        else:
            return ret_
    except KeyError as e:
        supported_names = ', '.join([f'{i}' for i in dict_.keys()])
        raise KeyError(f'No name {name}, only {supported_names} supported')


def get_optimizer(name):
    d = create_dict_from_module(
        torch.optim
    )
    name= name.lower()
    return get_(d, name)


def get_scheduler(name):
    d = create_dict_from_module(
        torch.optim.lr_scheduler 
    )
    name= name.lower()
    return get_(d, name)
          
    
def get_loss(name):
    d = create_dict_from_module(
        torch.nn
    )
    name= name.lower()
    return get_(d, name)
    

def get_transform(name):
    d = create_dict_from_module(
        torchvision.transforms
    )
    name= name.lower()
    return get_(d, name)
    
    
def get_model(name, source, **additional_params):
    if source.lower() in "pytorch":
        return get_torch_model(name, **additional_params)
    else:
        if name.lower() == "ofa":
            return get_ofa_model(name, **additional_params)
        else:
            return get_object(
                source, name)
    

def get_ofa_model(version, source):
    with open(source) as build_config_file:
        build_cfg = json.load(build_config_file)

    if version.lower() == "resnet":
        model = ResNets.build_from_config(build_cfg)
    elif version.lower() == "mobilenet":
        model = MobileNetVPlatComp.build_from_config(build_cfg)

    model_trace = torch.fx.symbolic_trace(model)
    def pattern(x):
        return x.view(x.size(0), -1)
    def replacement(x):
        return torch.flatten(x, start_dim=1)

    torch.fx.replace_pattern(model_trace, pattern, replacement)
    model = model_trace

    return model

    
def get_torch_model(name, version, classes):
    name = name.lower()
    version = version.lower()
    names=["resnet",'vgg']
    if name == "resnet":
        model =  torch.hub.load(
            'pytorch/vision:v0.10.0', 
            name+version, 
            pretrained=True
        )
        if classes is not None:
            in_f = model.fc.in_features
            model.fc = torch.nn.Linear(in_f, classes) 
        return model

    elif name == 'vgg':
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

    elif name == "mobilenet":
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

    
def get_dataset_creator():
    return generic_set_one_annotation


def get_object(source, name, version=None):

    mod = create_module_from_source(source)
    d = create_dict_from_module(mod)

    return get_(d, name, version)
