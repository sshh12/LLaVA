import torch.nn as nn
import re


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_match = re.match(
        r"^mlp_depth(\d+)_frames(\d+)$", config.mm_projector_type
    )

    mlp_depth = int(projector_match.group(1))
    frame_cnt = int(projector_match.group(2))
    modules = [nn.Linear(config.mm_hidden_size * frame_cnt, config.hidden_size)]
    for _ in range(1, mlp_depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(config.hidden_size, config.hidden_size))

    return nn.Sequential(*modules)
