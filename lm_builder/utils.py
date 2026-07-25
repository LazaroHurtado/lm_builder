from collections import OrderedDict
from functools import reduce

import torch
import yaml


def is_positive_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_yml(file):
    with open(file, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def select_positional_embedding(weight, x, position_ids=None):
    _, seq_len, embedding_dim = x.size()

    if position_ids is None:
        return weight[None, :seq_len, :embedding_dim]

    position_ids = position_ids.to(device=weight.device, dtype=torch.long)
    return weight[position_ids, :embedding_dim]


def module_has_attr(config, key, primary_module, fallback_module=None):
    if key in config and isinstance(config[key], str):
        if hasattr(primary_module, config[key]):
            config[key] = getattr(primary_module, config[key])
        elif (fallback_module is not None) and hasattr(fallback_module, config[key]):
            config[key] = getattr(fallback_module, config[key])
        else:
            raise AttributeError(f"Attribute not found {config[key]}")
    return config


def change_state_dict_names(
    original_state_dict: dict[str, torch.Tensor],
    name_changes: list[tuple[str, str]],
    to_transpose: list[str] = [],
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

        with torch.no_grad():
            new_state_dict[new_name] = parameters

    return new_state_dict


def combine_qkv_projections(state_dict: dict[str, torch.Tensor]):
    combined_state_dict = OrderedDict({})
    q_marker = ".attn.q_proj."
    k_marker = ".attn.k_proj."
    v_marker = ".attn.v_proj."

    for layer_name, parameters in state_dict.items():
        if q_marker in layer_name:
            prefix, parameter_name = layer_name.split(q_marker, 1)
            key_name = f"{prefix}{k_marker}{parameter_name}"
            value_name = f"{prefix}{v_marker}{parameter_name}"
            missing_names = [
                name for name in (key_name, value_name) if name not in state_dict
            ]
            if missing_names:
                raise KeyError(
                    f"Missing QKV projection tensors: {', '.join(missing_names)}"
                )

            combined_state_dict[f"{prefix}.attn.qkv_proj.{parameter_name}"] = torch.cat(
                (
                    parameters,
                    state_dict[key_name],
                    state_dict[value_name],
                ),
                dim=0,
            )
        elif k_marker in layer_name or v_marker in layer_name:
            marker = k_marker if k_marker in layer_name else v_marker
            prefix, parameter_name = layer_name.split(marker, 1)
            query_name = f"{prefix}{q_marker}{parameter_name}"
            if query_name not in state_dict:
                raise KeyError(f"{layer_name} has no matching query projection tensor.")
        else:
            combined_state_dict[layer_name] = parameters

    return combined_state_dict
