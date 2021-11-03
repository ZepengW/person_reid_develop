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
    use_loss = loss_cfg.get('use_loss',['cross_entropy'])
    logging.info(f'=> Initialing Loss Function: {use_loss}')
    for key in use_loss:
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
        loss_weight = loss_params.get('weight',1.0)
        loss_weight_l.append(loss_weight)
        logging.info(f'----loss:{key}')
        logging.info(f'------weight:{loss_weight}')
        loss_name_l.append(key)
        loss_input.append(loss_params.get('except_input'))
        weight_per_loss = loss_params.get('list_weight', [])
        weight_per_loss_l.append(weight_per_loss)
        logging.info(f'------list_weight(if need):{weight_per_loss}')

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
                # compute loss related with classify, such as cross entropy loss
                if isinstance(inputs, list):    # for list input
                    loss_value = torch.tensor(0.0).to(targets.device)
                    for j, input in enumerate(inputs):
                        loss_value_per = loss(input, targets)
                        loss_value += loss_value_per * weight_per_loss_l[i][j]
                        loss_value_l.append(float(loss_value_per.cpu()))
                        loss_name.append(loss_name_l[i]+f'_{j}')
                else:
                    loss_value = loss(inputs, targets)
                    loss_value_l.append(float(loss_value.cpu()))
                    loss_name.append(loss_name_l[i])
            elif loss_input[i] == 'feature':
                # compute loss related with feature distance, such as triplet loss
                if isinstance(feats, list):     # for list input
                    loss_value = torch.tensor(0.0).to(targets.device)
                    for j, feat in enumerate(feats):
                        loss_value_per = loss(feat, targets)
                        loss_value += loss_value_per * weight_per_loss_l[i][j]
                        loss_value_l.append(float(loss_value_per.cpu()))
                        loss_name.append(loss_name_l[i]+f'_{j}')
                else:
                    loss_value = loss(feats, targets)
                    loss_value_l.append(float(loss_value.cpu()))
                    loss_name.append(loss_name_l[i])

            total_loss += loss_weight_l[i] * loss_value
        return total_loss, loss_value_l, loss_name

    return loss_func