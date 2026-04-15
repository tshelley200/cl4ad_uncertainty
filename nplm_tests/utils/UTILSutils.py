import numpy as np
import json, torch

def json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist() if obj.ndim > 0 else obj.item()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj

def create_config_file(config_table, OUTPUT_DIRECTORY):
    with open('%s/config.json'%(OUTPUT_DIRECTORY), 'w') as outfile:
        json.dump(config_table, outfile, indent=4)
    return '%s/config.json'%(OUTPUT_DIRECTORY)

def generate_random_seed_from_time():
    seed = datetime.datetime.now().microsecond
    seed+= datetime.datetime.now().second
    seed+= datetime.datetime.now().minute
    return seed

def move_tensor_attrs_to_device(obj, device):
    for name, val in vars(obj).items():
        if torch.is_tensor(val):
            setattr(obj, name, val.to(device))
    return

def move_modules_to_device(model, device):
    # Move internal modules/objects you passed into model
    if hasattr(model, "novelty_finder") and model.novelty_finder is not None:
        model.novelty_finder = model.novelty_finder.to(device)
        move_tensor_attrs_to_device(model.novelty_finder, device)
        
    if hasattr(model, "norm_syst_finder") and model.norm_syst_finder is not None:
        model.norm_syst_finder = model.norm_syst_finder.to(device)
        move_tensor_attrs_to_device(model.norm_syst_finder, device)

    if hasattr(model, "shape_syst_finder_list") and model.shape_syst_finder_list is not None:
        for m in model.shape_syst_finder_list:
            m = m.to(device)
            move_tensor_attrs_to_device(m, device)
    return
