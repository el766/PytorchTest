import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

import time


from torch.nn import functional as F

import csv

from torch2trt import torch2trt


import deeplabv3

nb_classes=2

model=deeplabv3.Net()

w = int(1126/2)
h = int(1352/2)
bs = int(4)

input_data = torch.rand(bs,3, w, h).cuda()

model = model.cuda()
model.eval()

model.half()

start = time.time()
for i in range(50):
    input_data = torch.rand(bs,3, w, h).cuda().half()
    #start = time.time()
    outputs = model(input_data)
    #torch.cuda.synchronize()
#print(outputs)
print(("test %d" %i))
print(time.time()-start)


print("+++++++ trt +++++++++++")

input_sample = torch.rand(bs,3, w, h).cuda().half()
model_trt = torch2trt(model, [input_sample],max_batch_size=bs,fp16_mode=True)
torch.save(model_trt.state_dict(), 'trtModel.pth')

from torch2trt import TRTModule
start = time.time()
model_trt = TRTModule()

model_trt.load_state_dict(torch.load('trtModel.pth'))
print(time.time()-start)
print("------loaded--------")


start = time.time()
for i in range(50):
    input_data = torch.rand(bs,3, w, h).cuda().half()
    outputs = model_trt(input_data)
    torch.cuda.synchronize()
#print(outputs)
print(("test %d" %i))
print(time.time()-start)



#model.eval()
#print(model)
#print(model(input_data).size())




