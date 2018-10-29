# -*- coding:utf-8 -*-

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
import shutil
import random
import datetime
import os
import math
import h5py
import cv2
from PIL import Image
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
# np.set_printoptions(threshold='nan')
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from model.fcn import fcn
from model.unet import UNET_MODEL
from model.deeplabv3.model.deeplabv3 import DeepLabV3
from data_process.utils import label_mapping, RemoteSensingDataset
from data_process.deeplab_utils import add_weight_decay

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def img_transforms(img,label):
    # img, label = random_crop(img, label, crop_size)
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    img = transform(img)
    label = torch.from_numpy(label)
    return img, label

def predict(net, im, label): # 预测结果
    # cm = np.array(colormap).astype('uint8')
    with torch.no_grad():
        im = im.unsqueeze(0).cuda()
        out = net(im)
        pred = out.max(1)[1].squeeze().cpu().data.numpy()
        pred = label_mapping(pred)
    return pred, label_mapping(label.numpy())

    
if __name__=='__main__':
    
    val_dataset = RemoteSensingDataset(False, img_transforms)
    # val_dataset = RemoteSensingDataset(True, img_transforms)
    
    net = DeepLabV3() 
    net.load_state_dict(torch.load('./saved_model/deeplabv3/48.pkl'))
    net.cuda()   
    net.eval()
       
    # sample_list = random.sample(range(0,712), 6)
    # print(sample_list)
    
    for index in range(len(val_dataset)):
        test_data, test_label = val_dataset[index]
        pred, label = predict(net, test_data, test_label)
        
        plt.subplot(131)
        plt.axis('off')  
        plt.title('Original')        
        plt.imshow(Image.open(val_dataset.data_list[index]))
               
        plt.subplot(132)
        plt.axis('off')  
        plt.title('Truth')
        plt.imshow(label)  
       
        plt.subplot(133)
        plt.axis('off') 
        plt.title('Predict')
        plt.imshow(pred)       

        plt.savefig('/home/computer/lcy/pytorch/MyProject/Semantic-segmentation/predict_result/predict_' + str(index+1) +'.png')
        plt.close()
        print(index+1)