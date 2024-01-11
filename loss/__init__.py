from utils.logger import logger as logging
from .triplet_loss import TripletLoss, WeightedRegularizedTriplet, CrossEntropyLabelSmooth, SoftTriple
from .loss_multilabel import XentMultiLabel, XentMultiInput, TripletMultiInput, TripletMultiLabel, SoftlabelTripletMultiInput, XentMultiInputSoftLabel
import torch
from tabulate import tabulate

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



class LossManager(object):
    def __init__(self, loss_list):
        logging.info('Initial Loss Manager -- Begin')
        self.loss_func_d = dict()
        self.initial_losses(loss_list)
        logging.info('Initial Loss Manager -- Finish')

    def initial_losses(self, loss_list):
        tab_headers = ['Name', 'Type', 'Weight']
        tab_data = []

        for loss_args in loss_list:
            loss_name = loss_args['name']  # loss name for display
            loss_type = loss_args['type']  # select loss fuction from __factory_loss
            loss_kwargs = loss_args.get('kwargs', dict())  # params for initial loss module
            loss_weight = loss_args.get('weight', 1.0)  # weight of this loss fuction result
            expect_inputs = loss_args['expect_inputs']  # expected input to this loss function
            tab_data.append([loss_name, loss_type, loss_weight])
            if isinstance(expect_inputs, list):
                expect_inputs = {key: key for key in expect_inputs}  # convert list to dict
            func_loss = LossFactory[loss_type](**loss_kwargs)
            self.loss_func_d[loss_name] = [func_loss, loss_weight, expect_inputs]

        # for logging
        # 将表格格式化为字符串
        table_str = tabulate(tab_data, tab_headers, tablefmt="grid")
        # 按行打印表格
        for line in table_str.split('\n'):
            logging.info(line)

    def __call__(self, inputs):
        loss_total = 0
        loss_dict = {}
        for loss_name in self.loss_func_d.keys():
            func_loss, weight_loss, expect_inputs = self.loss_func_d[loss_name]
            # filter the expected inputs for loss function
            loss_input = {key: inputs[item] for key, item in expect_inputs.items()}
            loss_value = func_loss(**loss_input)
            loss_total += loss_value * weight_loss  # total loss
            loss_dict[loss_name] = loss_value.clone().detach().cpu()  # log single loss
        loss_dict['loss'] = loss_total
        return loss_dict
