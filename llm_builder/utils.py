import torch

from functools import reduce

def module_has_attr(config, key, primary_module, fallback_module=None):
    if key in config:
        if hasattr(primary_module, config[key]):
            config[key] = getattr(primary_module, config[key])
        elif (fallback_module is not None) and hasattr(fallback_module, config[key]):
            config[key] = getattr(fallback_module, config[key])
    
    return config

def change_state_dict_names(state_dict, original_state_dict, name_changes, to_transpose=[]):
    
    for name, parameters in original_state_dict.items():
            changes_to_make = [change for change in name_changes if change[0] in name]
            should_transpose = any(name.endswith(w) for w in to_transpose)

            new_name = reduce(lambda curr_name, change : curr_name.replace(change[0], change[1]),
                            changes_to_make,
                            name)
            if should_transpose:
                parameters = parameters.t()
            with torch.no_grad():
                state_dict[new_name].copy_(parameters)
    
    return state_dict