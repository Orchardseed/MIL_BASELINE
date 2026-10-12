import yaml
from addict import Dict
from ruamel.yaml import YAML
import ruamel.yaml as ryaml
import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)
    
def update_config_from_options(config, options):
    for option in options:
        key, value = option.split('=')
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = type(d[keys[-1]])(value)
    return config


def _merge_input_fields(base, supplied):
    for key, value in supplied.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            _merge_input_fields(base[key], value)
        else:
            base[key] = deepcopy(value)


def read_effective_yaml(yaml_path, input_path=None, options=None):
    config = read_yaml(yaml_path)
    source = Path(input_path) if input_path is not None else (
        Path(__file__).resolve().parents[1] / 'input.yaml'
    )
    shared = read_yaml(source)
    _merge_input_fields(config['Dataset'], shared.get('Dataset', {}))
    if options:
        update_config_from_options(config, options)
    return config


def write_effective_yaml(config, destination):
    plain = config.to_dict() if hasattr(config, 'to_dict') else deepcopy(config)
    with open(destination, 'x', encoding='utf-8') as stream:
        yaml.safe_dump(plain, stream, sort_keys=False, allow_unicode=True)


from ruamel.yaml import YAML

def change_yaml_by_options(yaml_path, options):
    Yaml = YAML(typ='rt') 
    with open(yaml_path, 'r', encoding='utf-8') as file:
        config = Yaml.load(file)  

    for option in options:
        key, value = option.split('=')
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = type(d[keys[-1]])(value)  

    with open(yaml_path, 'w', encoding='utf-8') as file:
        Yaml.dump(config, file) 



def save_yaml(args,yaml_path,options):
    dst_path = os.path.join(args.Logs.now_log_dir,os.path.basename(yaml_path))
    write_effective_yaml(args, dst_path)
