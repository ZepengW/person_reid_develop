from utils.logger import Logger
logging = Logger()
from .net.transreid import TransReID
from .net.transreid_ssl import TransReIDSSL
from .net.backbones.osnet import build_model as osnet

__factory_model = {
    'transreid': TransReID,
    'transreid_ssl': TransReIDSSL,
    'osnet': osnet
}


def make_model(model_name, model_params: dict):
    if not model_name in __factory_model.keys():
        logging.error(f'model : [{model_name}] is not defined')
    logging.info(f'=> Initialing Network:[{model_name}]')
    logging.info('----Network Params:')
    for k,v in model_params.items():
        logging.info(f'------{k}:{v}')
    return __factory_model[model_name](**model_params)
