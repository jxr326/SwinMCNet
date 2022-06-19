import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from net import SwinMCNet
from data import test_dataset
from options import opt
from collections import OrderedDict
import time
from os.path import splitext
from ptflops.flops_counter import get_model_complexity_info


#set device for test
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


#load the model
model = SwinMCNet()


base_weights = torch.load(opt.test_model)
new_state_dict = OrderedDict()
for k, v in base_weights.items():
    name = k[7:]   # remove 'module.'
    new_state_dict[name] = v 
model.load_state_dict(new_state_dict)

print('Loading base network...')


model.cuda()
model.eval()

#test
test_data_root = opt.test_data_root
maps_path = opt.maps_path

test_sets = ['VT5000/Test','VT1000','VT821']

for dataset in test_sets:

    save_path = maps_path + dataset + '/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dataset_path = test_data_root + dataset
    test_loader = test_dataset(dataset_path, opt.testsize)
    
    for i in range(test_loader.size):
        image, t, gt, (H, W), name = test_loader.load_data()
        image = image.cuda()
        t     = t.cuda()
        shape = (W,H)

        outi1, outt1, out1, outi2, outt2, out2 = model(image,t,shape)

        res = out2
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ',save_path + name)
        cv2.imwrite(save_path + name,res*255)


    print('Test Done!')