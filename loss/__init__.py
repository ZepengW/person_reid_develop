import logging
from .triplet_loss import TripletLoss, WeightedRegularizedTriplet, CrossEntropyLabelSmooth, SoftTriple
from .loss_multilabel import XentMultiLabel, XentMultiInput, TripletMultiInput, TripletMultiLabel, SoftlabelTripletMultiInput, XentMultiInputSoftLabel
import torch
import torch.nn as nn
LossFactory = {
    'cross_entropy': XentMultiInput,
    'xent': XentMultiInput,
    'softlabel_xent': XentMultiInputSoftLabel,
    'xent_multi_input': XentMultiInput,
    'xent_multi_label': XentMultiLabel,
    'xent_label_smooth': CrossEntropyLabelSmooth,
    'triplet': TripletMultiInput,
    'triplet_multi_label': TripletMultiLabel,
    'softlabel_triplet':SoftlabelTripletMultiInput,
    'wrt': WeightedRegularizedTriplet,
    'soft_triplet': SoftTriple,
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
        if not key in LossFactory.keys():
            logging.warning(f'Can Not Find Loss Function : {key}')
            continue
        loss_params = loss_cfg[key]
        params_initial = loss_cfg[key].get('params_initial', [])
        ## get initial param from params
        params_input = dict()
        for param_name in params_initial:
            if param_name in params.keys():
                params_input[param_name] = params[param_name]
        loss_func_l.append(LossFactory[key](**params_input, **loss_cfg[key].get('params', dict())))
        loss_weight = loss_params.get('weight',1.0)
        loss_weight_l.append(loss_weight)
        logging.info(f'----loss:{key}')
        logging.info(f'------weight:{loss_weight}')
        loss_name_l.append(key)
        loss_input.append(loss_params.get('except_input'))
        weight_per_loss = loss_params.get('list_weight', [])
        weight_per_loss_l.append(weight_per_loss)
        logging.info(f'------list_weight(if need):{weight_per_loss}')

    def loss_func(**kwargs):
        '''
        define loss function
        '''
        total_loss = []
        loss_value_l = []
        loss_name = []
        for i, loss in enumerate(loss_func_l):
            params_need = loss.__call__.__code__.co_varnames
            kwargs['input'] = kwargs['score']
            kwargs['feat'] = kwargs['feature']
            inputs = dict(filter(lambda x: x[0] in params_need, kwargs.items()))
            loss_value = loss(**inputs)
            loss_value_l.append(float(loss_value.cpu()))
            loss_name.append(loss_name_l[i])
            if torch.isinf(loss_value) or torch.isnan(loss_value):
                logging.warning(f'Appear Exception Loss: {loss_value}')
                continue
            total_loss.append(loss_weight_l[i] * loss_value)
        
        total_loss = sum(total_loss)
        return total_loss, loss_value_l, loss_name

    return loss_func