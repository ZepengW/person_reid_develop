import logging
from .triplet_loss import TripletLoss, WeightedRegularizedTriplet, CrossEntropyLabelSmooth

__factory_loss = {
    'xent_label_smooth': CrossEntropyLabelSmooth,
    'triplet': TripletLoss,
    'wrt': WeightedRegularizedTriplet
}


def make_loss(loss_cfg: dict, **params):
    loss_func_l = []
    loss_weight_l = []
    for key in loss_cfg.keys():
        if not key in __factory_loss.keys():
            logging.warning(f'Can Not Find Loss Function : {key}')
            continue
        loss_params = loss_cfg[key]
        params.update(loss_params)
        loss_func_l.append(__factory_loss[key](**params))
        loss_weight_l.append(loss_params.get('weight',1.0))

    def loss_func(inputs, feats, targets):
        '''
        define loss function
        :param inputs: output of classify layer
        :param feats: features
        :param targets: labels
        :return:
        '''
        total_loss = 0.0
        for i, loss in enumerate(loss_func_l):
            loss_value = loss(inputs=inputs,feats=feats,targets=targets)
            if isinstance(loss_value,tuple):    # triplet loss return 3-tuple
                loss_value = loss_value[0]
            total_loss += loss_weight_l[i] * loss_value
        return total_loss

    return loss_func