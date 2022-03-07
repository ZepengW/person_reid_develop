import logging
from .net.resnet import resnet50
from .net.joint_transformer import JointFromer,JointFromerPCB,OnlyPCB,JointFromerV0_6,JointFromerPCBv2,JointFromerPCBv3
from .net.joint_transformer import JointFromerPCBv4
from .net.transreid import TransReID
from .net.maskformer import MaskFormer, MaskFormer2
from .net.simmim import SimMIM, SimMIMFinetune

__factory_model = {
    'resnet50': resnet50,
    'jointformer': JointFromer,
    'transreid': TransReID,
    'jointformer_pcb': JointFromerPCB,
    'jointformer_pcb_v2': JointFromerPCBv2,
    'jointformer_pcb_v3': JointFromerPCBv3,
    'jointformer_pcb_v4': JointFromerPCBv4,
    'only_pcb': OnlyPCB,
    'jointformerv0.6': JointFromerV0_6,
    'maskformer': MaskFormer,
    'maskformer2': MaskFormer2,
    'simmim': SimMIM,
    'simmim_finetune': SimMIMFinetune
}


def make_model(model_name, model_params: dict):
    if not model_name in __factory_model.keys():
        logging.error(f'model : [{model_name}] is not defined')
    logging.info(f'=> Initialing Network:[{model_name}]')
    logging.info('----Network Params:')
    for k,v in model_params.items():
        logging.info(f'------{k}:{v}')
    return __factory_model[model_name](**model_params)
