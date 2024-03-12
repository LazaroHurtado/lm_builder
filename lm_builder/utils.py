import torch

from collections import defaultdict
from functools import reduce

def module_has_attr(config, key, primary_module, fallback_module=None):
    if key in config:
        if hasattr(primary_module, config[key]):
            config[key] = getattr(primary_module, config[key])
        elif (fallback_module is not None) and hasattr(fallback_module, config[key]):
            config[key] = getattr(fallback_module, config[key])
        else:
            raise AttributeError(f"Attribute not found {config[key]}")
    return config

def merge_qkv_projs(q_proj_weight, k_proj_weight, v_proj_weight):
    qkv_proj_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
    return qkv_proj_weight

def merge_state_dict_qkv_projs(state_dict, q_proj, k_proj, v_proj, new_proj_name):
    block_projs = defaultdict(dict)
    proj_names = [(q_proj, "q_proj_weight"),
                  (k_proj, "k_proj_weight"),
                  (v_proj, "v_proj_weight")]

    for layer_name, parameters in state_dict.items():
        for proj_name, proj_key in proj_names:
            if proj_name in layer_name:
                layer_prefix, layer_suffix = layer_name.split(proj_name)
                block_projs[(layer_prefix, layer_suffix)][proj_key] = parameters
    
    for (layer_prefix, layer_suffix), parameters in block_projs.items():
        new_layer_name = layer_prefix+new_proj_name+layer_suffix
        state_dict[new_layer_name] = merge_qkv_projs(**parameters)
        
        for proj_name, _ in proj_names:
            old_layer_name = layer_prefix+proj_name+layer_suffix
            del state_dict[old_layer_name]

    return state_dict

def change_state_dict_names(state_dict, original_state_dict, name_changes, to_transpose=[]):
    
    for layer_name, parameters in original_state_dict.items():
            changes_to_make = [change for change in name_changes if (change[0] in layer_name)]
            should_transpose = any(layer_name.endswith(w) for w in to_transpose)

            new_name = reduce(lambda curr_name, change : curr_name.replace(change[0], change[1]),
                              changes_to_make,
                              layer_name)
            if should_transpose:
                parameters = parameters.t()
            with torch.no_grad():
                state_dict[new_name].copy_(parameters)
    
    return state_dict