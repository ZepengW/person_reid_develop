import torch
from utils.logger import Logger
logging = Logger()
import re
from collections import defaultdict

def make_optimizer(cfg_solver:dict, model, center_criterion= None):
    """
    generate optimizer from cfg_solver
    :param cfg_solver: dict
    :param model: torch.nn.Module
    :param center_criterion:
    :return: optimizer, optimizer_center
    """
    optimizer = cfg_solver.get('optimizer', 'Adam')
    lr_base = cfg_solver.get('base_lr', 0.0003)
    weight_decay_base = cfg_solver.get('weight_decay', 0.0005)
    logging.info(f'=> Initial Optimizer:{optimizer}')
    logging.info(f'----base base_lr:{lr_base}')
    logging.info(f'----base weight_decay:{weight_decay_base}')
    cfg_params = cfg_solver.get('params_group', dict())
    spec_layer = cfg_solver.get('spec_layer', [])
    #format [{'re':'layer_name regex', 'lr':'lr', 'weight_decay':'weight_decay'}]
    # if lr is 0, demonstrate the layers matched is frozen
    params_groups = defaultdict(list)
    params_groups['default'] = {"params": [], "group_name": "default"}
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params_groups['default']['params'].append(value)
    params = list(params_groups.values())
    if optimizer == 'SGD':
        optimizer = getattr(torch.optim, optimizer)(params, momentum=cfg_solver.get('momentum', 0.9), lr=lr_base,
                                      weight_decay=weight_decay_base)
    elif optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=lr_base,
                                      weight_decay=weight_decay_base)
    else:
        optimizer = getattr(torch.optim, optimizer)(params, lr = lr_base, weight_decay = weight_decay_base)

    if not center_criterion is None:
        optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr= cfg_solver.get('center_lr', 0.5))
    else:
        optimizer_center = None

    return optimizer, optimizer_center
