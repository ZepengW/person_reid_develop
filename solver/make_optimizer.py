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
    lr_base = cfg_solver.get('lr', 0.0003)
    weight_decay_base = cfg_solver.get('weight_decay', 0.0005)
    logging.info(f'=> Initial Optimizer:{optimizer}')
    logging.info(f'----base base_lr:{lr_base}')
    logging.info(f'----base weight_decay:{weight_decay_base}')
    # if lr is 0, demonstrate the layers matched is frozen
    params_groups = []
    params_groups.append({"key":'', "group_name": "default",
                                "params": [], "params_name": [],
                                "lr": lr_base, "weight_decay": weight_decay_base})
    cfg_lrgroup = cfg_solver.get('lr_group')
    for lr_group in cfg_lrgroup:
        key = [lr_group['key']] if isinstance(lr_group['key'], str) else lr_group['key']
        lr = lr_group.get('lr', lr_base)
        weight_decay = lr_group.get('weight_decay', weight_decay_base)
        params_groups.append({
            'key': key,
            'group_name': lr_group['name'],
            'params': [],
            'params_name': [],
            'lr': lr,
            'weight_decay': weight_decay
        })

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        flag_finish = False
        for item in params_groups[1:]:
            for key_search in item['key']:
                if re.search(key_search, key):
                    if item['lr'] == 0:
                        value.requires_grad = False
                    item['params'].append(value)
                    item['params_name'].append(key)
                    flag_finish = True
                    break
        if not flag_finish:
            params_groups[0]['params'].append(value)
            params_groups[0]['params_name'].append(key)
    log_network_lr(params_groups)
    if optimizer == 'SGD':
        optimizer = getattr(torch.optim, optimizer)(params_groups, momentum=cfg_solver.get('momentum', 0.9))
    elif optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_groups)
    else:
        optimizer = getattr(torch.optim, optimizer)(params_groups)

    if not center_criterion is None:
        optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr= cfg_solver.get('center_lr', 0.5))
    else:
        optimizer_center = None

    return optimizer, optimizer_center


def log_network_lr(params_groups):
    """
    Log the learning rate and weight decay of each parameter group in a beautiful manner
    :param params_groups: dict
    """
    for group in params_groups:
        lr = group['lr']
        weight_decay = group['weight_decay']
        logging.info(f"Group: {group['group_name']}, Key: {group['key']} Initial lr: {lr:.2e}, Initial weight decay: {weight_decay:.2e}")
        for param_name in group['params_name']:
            logging.info(f"----Parameter: {param_name}")