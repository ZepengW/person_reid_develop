import torch.nn as nn
import torch
from model.net.resnet import resnet50, resnet101

class MbNetwork(nn.Module):

    def __init__(self, pretrained=True):
        super(MbNetwork, self).__init__()
        self.model_body = resnet101(pretrained=pretrained)
        self.model_clothes = resnet50(pretrained=pretrained)
        #self.model_contour = resnet50(pretrained=pretrained)
        self.model_total = resnet101(pretrained=pretrained)
        # fuse features by layer
        #self.layer_fuse =  

    def forward(self,x,masks):
        # x : (batch, body_part_channel, c, h, w)
        _, f_whole = self.model_total(x)

        x = torch.einsum('nchw,nbhw->nbchw', x, masks)
        #
        x_body = x[:,0,:,:,:]
        x_clothes = x[:,1,:,:,:]
        input_contour = masks[:,2,:,:]
        input_contour = 255 * input_contour
        b,h,w = input_contour.shape
        input_contour = input_contour.view(b,1,h,w)
        input_contour = input_contour.repeat(1,3,1,1)
        _, f_body = self.model_body(x_body)
        _, f_clothes = self.model_clothes(x_clothes)
        #_, f_contour = self.model_contour(input_contour)
        f_total = torch.cat((f_whole,f_body),dim=1)
        return f_total
