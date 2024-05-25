import torch

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


def partition_attn(parameters: torch.Tensor):
    q_proj, k_proj, v_proj = parameters.chunk(3, dim=0)

    return q_proj, k_proj, v_proj


def change_state_dict_names(
    original_state_dict: dict[str, torch.Tensor],
    name_changes: list[tuple[str, str]],
    to_transpose: list[str] = [],
    to_partition: str = None,
    remove_bias: bool = False,
):
    new_state_dict = OrderedDict({})

    for layer_name, parameters in original_state_dict.items():
        if remove_bias and layer_name.endswith("bias"):
            continue

        changes_to_make = [
            change for change in name_changes if (change[0] in layer_name)
        ]
        should_transpose = any(layer_name.endswith(w) for w in to_transpose)

        new_name = reduce(
            lambda curr_name, change: curr_name.replace(change[0], change[1]),
            changes_to_make,
            layer_name,
        )
        if should_transpose:
            parameters = parameters.t()

        if to_partition is not None and to_partition in layer_name:
            parameters = partition_attn(parameters)
            new_names = [
                new_name.replace(to_partition, ".q_proj."),
                new_name.replace(to_partition, ".k_proj."),
                new_name.replace(to_partition, ".v_proj."),
            ]
        else:
            parameters = [parameters]
            new_names = [new_name]

        with torch.no_grad():
            for new_name, new_parameter in zip(new_names, parameters):
                new_state_dict[new_name] = new_parameter

    return new_state_dict
