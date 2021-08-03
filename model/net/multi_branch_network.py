import torch.nn as nn
import torch
from model.net.resnet import resnet50

class MbNetwork(nn.Module):

    def __init__(self, pretrained=True):
        super(MbNetwork, self).__init__()
        self.model_body = resnet50(pretrained=pretrained)
        self.model_clothes = resnet50(pretrained=pretrained)

    def forward(self,x,masks):
        # x : (batch, body_part_channel, c, h, w)
        x = torch.einsum('nchw,nbhw->nbchw', x, masks)
        #
        x_body = x[:,0,:,:,:]
        x_clothes = x[:,1,:,:,:]
        _,f_body = self.model_body(x_body)
        _,f_clothes = self.model_clothes(x_clothes)
        f_total = 0.8*f_body+0.2*f_clothes
        return f_total
