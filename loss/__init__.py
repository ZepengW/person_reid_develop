import logging
from .triplet_loss import TripletLoss, WeightedRegularizedTriplet, CrossEntropyLabelSmooth
import torch
import torch.nn as nn
__factory_loss = {
    'cross_entropy': nn.CrossEntropyLoss,
    'xent_label_smooth': CrossEntropyLabelSmooth,
    'triplet': TripletLoss,
    'wrt': WeightedRegularizedTriplet
}


def make_loss(loss_cfg: dict, **params):
    loss_func_l = []
    loss_weight_l = []
    loss_name_l = []
    loss_input = []
    weight_per_loss_l = []
    for key in loss_cfg.get('use_loss',['cross_entropy']):
        if not key in __factory_loss.keys():
            logging.warning(f'Can Not Find Loss Function : {key}')
            continue
        loss_params = loss_cfg[key]
        params_initial = loss_cfg[key].get('params_initial', [])
        ## get initial param from params
        params_input = dict()
        for param_name in params_initial:
            if param_name in params.keys():
                params_input[param_name] = params[param_name]
        loss_func_l.append(__factory_loss[key](**params_input))
        loss_weight_l.append(loss_params.get('weight',1.0))
        loss_name_l.append(key)
        loss_input.append(loss_params.get('except_input'))
        weight_per_loss_l.append(loss_params.get('list_weight', []))

    def loss_func(inputs, feats, targets):
        '''
        define loss function
        :param inputs: output of classify layer
        :param feats: features
        :param targets: labels
        :return: total_loss, loss_value:list, loss_name:list
        '''
        total_loss = 0.0
        loss_value_l = []
        loss_name = []
        for i, loss in enumerate(loss_func_l):
            if loss_input[i] == 'score':
                if isinstance(inputs, list):
                    loss_value = torch.tensor(0.0).to(targets.device)
                    for j, input in enumerate(inputs):
                        loss_value += loss(input, targets) * weight_per_loss_l[i][j]
                    loss_value = loss_value / len(inputs)
                else:
                    loss_value = loss(inputs, targets)
            elif loss_input[i] == 'feature':
                if isinstance(feats, list):
                    loss_value = torch.tensor(0.0).to(targets.device)
                    for j, feat in enumerate(feats):
                        loss_value += loss(feat, targets) * weight_per_loss_l[i][j]
                    loss_value = loss_value / len(inputs)
                else:
                    loss_value = loss(feats, targets)
            total_loss += loss_weight_l[i] * loss_value
            loss_value_l.append(float(loss_value.cpu()))
            loss_name.append(loss_name_l[i])
        return total_loss, loss_value_l, loss_name

    return loss_func