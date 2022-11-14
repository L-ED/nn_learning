import torch, torchvision, os, sys, json
from ..datasets import generic_set_one_annotation

import yaml
sys.path.insert(0, "/storage_labs/3030/LyginE/projects/paradigma/platform_converter/models")
from platform_converter.models.models.networks.classification.ofa_mobilenetv3 import MobileNetVPlatComp
from platform_converter.models.models.networks.classification.ofa_resnet import ResNets


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
    
    
def get_model(name, version, source):
    if source.lower() in "pytorch":
        return get_torch_model(name, version)
    else:
        if name.lower() == "ofa":
            return get_ofa_model(
                version, source)
        else:
            return get_object(
                source, name, version)
    

def get_ofa_model(version, source):
    with open(source) as build_config_file:
        build_cfg = json.load(build_config_file)

    if version.lower() == "resnet":
        return ResNets.build_from_config(build_cfg)
    elif version.lower() == "mobilenet":
        return MobileNetVPlatComp.build_from_config(build_cfg)

    
def get_torch_model(name, version):
    name = name.lower()
    version = version.lower()
    if name == "resnet":
        return torch.hub.load(
            'pytorch/vision:v0.10.0', 
            name+version, 
        )
    if name == "mobilenet":
        return torch.hub.load(
            'pytorch/vision:v0.10.0', 
            name + "_" + version)

    
def get_dataset_creator():
    return generic_set_one_annotation


def get_object(source, name, version=None):

    print(source)
    mod = create_module_from_source(source)
    d = create_dict_from_module(mod)

    return get_(d, name, version)
