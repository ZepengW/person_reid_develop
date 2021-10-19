import torch


def make_optimizer(cfg_solver:dict, model, center_criterion= None):
    """
    generate optimizer from cfg_solver
    :param cfg_solver: dict
    :param model: torch.nn.Module
    :param center_criterion:
    :return: optimizer, optimizer_center
    """
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg_solver.get('base_lr', 0.0003)
        weight_decay = cfg_solver.get('weight_decay', 0.0005)
        if "bias" in key:
            lr = lr * cfg_solver.get('bias_lr_factor', 0.0003)
            weight_decay = cfg_solver.get('weight_decay_bias', 0.0005)
        if cfg_solver.get('large_fc_lr', False):
            # Whether using larger learning rate for fc layer
            if "classifier" in key or "arcface" in key:
                lr = lr * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = cfg_solver.get('optimizer', 'Adam')
    if optimizer == 'SGD':
        optimizer = getattr(torch.optim, optimizer)(params, momentum=cfg_solver.get('momentum', 0.9))
    elif optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg_solver.get('base_lr', 0.0003),
                                      weight_decay=cfg_solver.get('weight_decay', 0.0005))
    else:
        optimizer = getattr(torch.optim, cfg_solver.get('base_lr', 0.0003))(params)

    if not center_criterion is None:
        optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr= cfg_solver.get('center_lr', 0.5))
    else:
        optimizer_center = None

    return optimizer, optimizer_center
