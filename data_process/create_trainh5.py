#coding:utf-8
#训练数据读取 train.h5

import os
from PIL import Image
import numpy as np
import scipy.io
import torchvision.transforms as transforms
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

transform = transforms.Compose([
        # transforms.Resize((384,128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ]
    )

train_img_path = '/home/computer/lcy/pytorch/MyProject/Semantic-segmentation/data_process/new_dataset/'

all_img = []
all_label = []

filename = os.listdir(train_img_path)
for name in filename:
    if(name.find('_class') >=  0):
        continue
    elif(name.find('_color') >= 0):
        continue
    else:
        


for i in range(13178):   
    train_img_label = train_data[i][1][0][0]
    
    if train_img_label == temp_label:
        temp_label = train_img_label
        train_img_label = num
    else:
        temp_label = train_img_label
        num += 1
        train_img_label = num
    
    temp_img_path = train_data[i][0][0]
    img = Image.open(os.path.join(train_img_path,str(temp_img_path)))
    img = transform(img)
    img = img.numpy()
    img = img.tolist()
    print(temp_img_path, train_img_label)
    print('------------------------------------------')
    
    all_img.append(img)
    all_label.append(train_img_label)
    
train_img = np.asarray(all_img,dtype=np.float32)   
print(train_img.shape)
train_label = np.asarray(all_label,dtype=np.int64)  
print(train_label.shape)  

f = h5py.File('train_384_128_13178.h5','w')
f['train_img'] = train_img                
f['train_label'] = train_label
f.close()
