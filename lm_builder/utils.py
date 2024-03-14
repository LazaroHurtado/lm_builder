import torch
import gc

from collections import OrderedDict
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

    del q_proj_weight, k_proj_weight, v_proj_weight
    gc.collect()

    return qkv_proj_weight


def merge_state_dict_qkv_projs(state_dict, q_proj, k_proj, v_proj, new_proj_name):
    state_dict[new_proj_name] = merge_qkv_projs(
        state_dict[q_proj], state_dict[k_proj], state_dict[v_proj]
    )

    del state_dict[q_proj]
    del state_dict[k_proj]
    del state_dict[v_proj]
    gc.collect()

    return state_dict


def change_state_dict_names(original_state_dict, name_changes, to_transpose=[]):
    new_state_dict = OrderedDict({})

    for layer_name, parameters in original_state_dict.items():
        changes_to_make = [
            change for change in name_changes if (change[0] in layer_name)
        ]
        should_transpose = any((w in layer_name) for w in to_transpose)

        new_name = reduce(
            lambda curr_name, change: curr_name.replace(change[0], change[1]),
            changes_to_make,
            layer_name,
        )
        if should_transpose:
            parameters = parameters.t()
        with torch.no_grad():
            new_state_dict[new_name] = parameters
            del parameters
            gc.collect()

    return new_state_dict
