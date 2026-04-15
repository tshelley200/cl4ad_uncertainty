import numpy as np
import json

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