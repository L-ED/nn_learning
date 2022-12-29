import torch, torchvision, os, sys
from ..datasets import DATASET_CREATORS
from ..models import MODELS, MODEL_POSTRPOCESSORS

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


def get_object(source, name, version=None):

    mod = create_module_from_source(source)
    d = create_dict_from_module(mod)

    return get_(d, name, version)


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
    

def get_dataset_creator(task):
    return DATASET_CREATORS[task]

    
def get_model(name, source, **additional_params):
    source = source.lower()
    
    if "pytorch" in source:
        return get_torch_model(name, **additional_params)
    
    elif "nnl" in source:
        return get_nnl_model(name, **additional_params)
    
    elif "file" in source:
        return get_object(
                source, name)

    else:
        raise ValueError("No source mode {}. Only 'pytorch' 'nnl' 'file' supported")


def get_nnl_model(name, **postprocessing_params):
    return MODELS[name](**postprocessing_params)

    
def get_torch_model(name, **postprocessing_params):
    name = name.lower()
    
    models_dict = create_dict_from_module(
        torchvision.models)
    model = get_(
        models_dict, name)
    
    model = MODEL_POSTRPOCESSORS[name](
        model, **postprocessing_params)
    
    return model

