# provide loss function for two label merge
import torch.nn as nn
import torch
from utils.logger import Logger
logging = Logger()
from .triplet_loss import TripletLoss, SoftLabelTriplet

class XentMultiInput(object):
    """Xent loss function, which has a list of feature as input

    Args:
        object (_type_): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self, list_weight=[1.0]):
        self.xent = nn.CrossEntropyLoss()
        self.weights = list_weight
    
    def __call__(self, input, target):
        # check input
        if type(input) is not list:
            input = [input]
        if len(input) > len(self.weights):
            logging.error(f'inputs length[{len(input)}] is longer than the except [{len(self.weights)}]')
        loss_value = [self.xent(input[i], target) * self.weights[i] for i in range(len(input))]
        return sum(loss_value)

class XentMultiInputSoftLabel(object):
    """Xent loss function, which has a list of feature as input

    Args:
        object (_type_): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self, list_weight=[1.0]):
        self.xent = nn.CrossEntropyLoss()
        self.weights = list_weight
    
    def __call__(self, input, soft_target):
        # check input
        if type(input) is not list:
            input = [input]
        if len(input) > len(self.weights):
            logging.error(f'inputs length[{len(input)}] is longer than the except [{len(self.weights)}]')
        loss_value = [self.xent(input[i], soft_target) * self.weights[i] for i in range(len(input))]
        return sum(loss_value)

class XentMultiLabel(object):
    # for mixing strategy
    def __init__(self, list_weight=[1.0]):
        self.xent = nn.CrossEntropyLoss()
        self.weights = list_weight
        
    
    def __call__(self, input, target_l, weight_l):
        if type(input) is not list:
            input = [input]
        if len(input) > len(self.weights):
            logging.error(f'inputs length[{len(input)}] is longer than the except [{len(self.weights)}]')
        loss_value = []
        for j in range(len(input)):
            loss_value.append(sum([self.xent(input[j], target_l[i]) * torch.mean(weight_l[i]) for i in range(len(target_l))]) * self.weights[j])

        return sum(loss_value)



class TripletMultiInput(object):
    """Triplet loss function, which has a list of feature as input

    Args:
        object (_type_): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self, margin=None, hard_factor=0.0, list_weight=[1.0]):
        self.trip = TripletLoss(margin=margin, hard_factor=hard_factor)
        self.weights = list_weight
    
    def __call__(self, feat, target):
        # check input
        if type(feat) is not list:
            feat = [feat]
        if len(feat) > len(self.weights):
            logging.error(f'inputs length[{len(feat)}] is longer than the except [{len(self.weights)}]')
        loss_value = [self.trip(feat[i], target) * self.weights[i] for i in range(len(feat))]
        return sum(loss_value)


class TripletMultiLabel(object):
    # for mixing strategy
    def __init__(self, margin=None, hard_factor=0.0, list_weight=[1.0]):
        self.trip = TripletLoss(margin=margin, hard_factor=hard_factor)
        self.weights = list_weight
    
    def __call__(self, feat, target_l, weight_l):
        if type(feat) is not list:
            feat = [feat]
        if len(feat) > len(self.weights):
            logging.error(f'inputs length[{len(feat)}] is longer than the except [{len(self.weights)}]')
        loss_value = []
        for j in range(len(feat)):
            loss_value.append(sum([self.trip(feat[j], target_l[i]) * torch.mean(weight_l[i]) for i in range(len(target_l))]) * self.weights[j])
        return sum(loss_value)


class SoftlabelTripletMultiInput(object):
    """Triplet loss function, which has a list of feature as input

    Args:
        object (_type_): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self, margin=None, hard_factor=0.0, list_weight=[1.0], thred = 0.5):
        logging.info("====> Initial [SoftlabelTripletMultiInput]")
        logging.info(f"======> list_weight={list_weight}")
        logging.info(f"======> thred={thred}")
        self.trip = SoftLabelTriplet(margin=margin, hard_factor=hard_factor, thred = thred)
        self.weights = list_weight
    
    def __call__(self, feat, soft_target):
        # check input
        if type(feat) is not list:
            feat = [feat]
        if len(feat) > len(self.weights):
            logging.error(f'inputs length[{len(feat)}] is longer than the except [{len(self.weights)}]')
        loss_value = [self.trip(feat[i], soft_target) * self.weights[i] for i in range(len(feat))]
        return sum(loss_value)
