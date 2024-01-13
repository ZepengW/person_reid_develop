""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler
import torch.optim.lr_scheduler as LrS 
from utils.logger import Logger
logging = Logger()

def create_scheduler(cfg_solver, optimizer):
    scheduler_type = cfg_solver.get('scheduler_type', 'cosine').lower()
    if scheduler_type == 'cosine':
        num_epochs = cfg_solver.get('max_epoch', 120)
        # type 1
        # lr_min = 0.01 * cfg.SOLVER.BASE_LR
        # warmup_lr_init = 0.001 * cfg.SOLVER.BASE_LR
        # type 2
        lr_min = 0.002 * cfg_solver.get('base_lr', 0.0003)
        warmup_lr_init = 0.01 * cfg_solver.get('base_lr', 0.0003)
        # type 3
        # lr_min = 0.001 * cfg.SOLVER.BASE_LR
        # warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR

        warmup_t = cfg_solver.get('warmup_epochs', 5)
        noise_range = None
        input_params = {
            "t_initial": num_epochs,
            "lr_min":lr_min,
            "t_mul": 1.,
            "decay_rate":0.1,
            "warmup_lr_init":warmup_lr_init,
            "warmup_t":warmup_t,
            "cycle_limit":1,
            "t_in_epochs":True,
            "noise_range_t":noise_range,
            "noise_pct": 0.67,
            "noise_std": 1.,
            "noise_seed":42
        }
        lr_scheduler = CosineLRScheduler(
                optimizer,
                **input_params
            )
    elif scheduler_type == 'step':
        input_params = {
            "step_size": cfg_solver.get('step_size', 10),
            "gamma": cfg_solver.get('gamma', 0.1),
            "last_epoch": cfg_solver.get('max_epoch', -1)
        }
        lr_scheduler = LrS.StepLR(
            optimizer=optimizer,
            **input_params
        )
    elif scheduler_type == 'multi-step':
        input_params = {
            "step_size": cfg_solver.get('step_size', 10),
            "milestones": cfg_solver.get('milestones', []),
            "gamma": cfg_solver.get('gamma', 0.1),
            "last_epoch": cfg_solver.get('max_epoch', -1)
        }
        lr_scheduler = LrS.MultiStepLR(
            optimizer=optimizer,
            **input_params
        )
    else:
        logging.error(f'UNSUPPORT [scheduler_type] : {scheduler_type}')
        raise(f'UNSUPPORT [scheduler_type] : {scheduler_type}')
    logging_scheduler(scheduler_type, input_params)
    return lr_scheduler

def logging_scheduler(scheduler_type, input_params:dict):
    logging.info(f'=> Initial LR Scheduler:{scheduler_type}')
    for (key, value) in input_params.items():
        logging.info(f'----{key}:{value}')
