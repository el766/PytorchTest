import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
#from resnet50 import Resnet50 as backbone
import resnet

from torchvision.models.segmentation.deeplabv3 import DeepLabHead as classifier


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.backbone=resnet.resnet50(replace_stride_with_dilation = [False, True, True])
        self.classifier = classifier(2048,2)




    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.backbone(x)
        x = self.classifier(x)
        return x


if __name__=="__main__":
    net = Net()
    print(net)

