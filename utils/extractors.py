import yaml

def dict_from_yaml(filepath):
    with open(filepath, 'r') as config:
        data= yaml.safe_load(config)
        
    return data