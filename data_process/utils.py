# -*- coding:utf-8 -*-

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
from PIL import Image
import numpy as np
np.set_printoptions(threshold=1e100)
import torchvision.transforms as transforms
import torchvision
import torchvision.models as models
import torch
import torch.nn as nn
import torch.utils.data as data

def read_images(train=True):   
    txt_fname = '/home/computer/lcy/pytorch/MyProject/Semantic-segmentation/data_process/new_dataset/' + ('train.txt' if train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    data = [os.path.join('/home/computer/lcy/pytorch/MyProject/Semantic-segmentation/data_process/new_dataset/image', i+'.png') for i in images]
    label = [os.path.join('/home/computer/lcy/pytorch/MyProject/Semantic-segmentation/data_process/new_dataset/label', i+'.png') for i in images]
    return data, label

 
def rand_crop(data, label, height, width):
    '''
    data is PIL.Image object
    label is PIL.Image object
    '''
    data, rect = transforms.RandomCrop((height, width))(data)
    label = transforms.FixedCrop(*rect)(label)
    return data, label
    

def image2label(im):
    
    colormap = [[0, 255, 0], [255, 0, 0], [0, 0, 255], [190, 190, 190], [255, 255, 255]] #  绿色  红色  蓝色  灰色  白色 
    
    cm2lbl = np.zeros(256**3) # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
    for i,cm in enumerate(colormap):
        cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i # 建立索引
    data = np.array(im, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64') # 根据索引得到 label 矩阵  


def label_mapping(label_im):
    colorize = np.zeros([5,3],dtype=np.int64)
    colorize[0,:] = [255, 255, 255]     # 其它  白色
    colorize[1,:] = [0, 255, 0]         # 植被  绿色
    colorize[2,:] = [190, 190, 190]     # 道路  灰色
    colorize[3,:] = [255, 0, 0]         # 建筑  红色
    colorize[4,:] = [0, 0, 255]         # 水体  蓝色
    
    label = colorize[label_im,:].reshape([label_im.shape[0], label_im.shape[1], 3])
    return label


class RemoteSensingDataset(data.Dataset):

    def __init__(self, train, transforms):
        self.transforms = transforms
        self.data_list, self.label_list = read_images(train=train)
        print('Read ' + str(len(self.data_list)) + ' images')
        
    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = Image.open(img)
        label = Image.open(label)
        label = np.asarray(label, dtype=np.int64)
        img, label = self.transforms(img, label)
        return img, label
    
    def __len__(self):
        return len(self.data_list)
    
# if __name__=='__main__':
    # label_im = Image.open('/home/computer/lcy/pytorch/MyProject/Semantic-segmentation/data_process/new_dataset/label/1_1000.png')
    # label_im = np.asarray(label_im, dtype=np.int64)
    # print(np.sum(label_im==0))
    # print(np.sum(label_im==1)) 
    # print(np.sum(label_im==2))
    # print(np.sum(label_im==3))
    # print(np.sum(label_im==4))
    # label_im = label_mapping(label_im)
    # plt.imshow(label_im)
    # plt.savefig('1.png')
    
    # -------------------------------------------------------------------------------------------------------------
    # train_dataset = RemoteSensingDataset(True, img_transforms)
    # val_dataset = RemoteSensingDataset(False, img_transforms)

    # train_data = data.DataLoader(train_dataset, 64, shuffle=True, num_workers=4)
    # val_data = data.DataLoader(val_dataset, 128, num_workers=4)

    