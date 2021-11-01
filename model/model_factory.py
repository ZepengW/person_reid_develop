import logging
from .net.resnet import resnet50
from .net.joint_transformer import JointFromer,JointFromerPCB
from .net.transreid import TransReID

__factory_model = {
    'resnet50': resnet50,
    'jointformer': JointFromer,
    'transreid': TransReID,
    'jointformer_pcb': JointFromerPCB
}


def make_model(model_name, model_params: dict):
    if not model_name in __factory_model.keys():
        logging.error(f'model : [{model_name}] is not defined')
    return __factory_model[model_name](**model_params)
