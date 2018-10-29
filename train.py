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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc    
    
if __name__=='__main__':

    train_dataset = RemoteSensingDataset(True, img_transforms)
    val_dataset = RemoteSensingDataset(False, img_transforms)

    train_data = data.DataLoader(train_dataset, 64, shuffle=True, num_workers=4)
    val_data = data.DataLoader(val_dataset, 128, num_workers=4)
    
    # net = fcn(5)
    # net.load_state_dict(torch.load('./saved_model/1/8.pkl'))
    # net = UNET_MODEL()
    
    net = DeepLabV3()   
    net = net.cuda()
    
    criterion = nn.NLLLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, weight_decay=1e-4)   
    
    params = add_weight_decay(net, l2_value=0.0001)
    optimizer = torch.optim.Adam(params, lr=1e-3)
    
    for e in range(1000):
        # if e > 0 and e % 50 == 0:
            # optimizer.set_learning_rate(optimizer.learning_rate * 0.1)
            
        net = net.train()
        train_loss = 0
        train_acc = 0
        train_acc_cls = 0
        train_mean_iu = 0
        train_fwavacc = 0
        
        prev_time = datetime.datetime.now()
        for data in train_data:
            im = data[0].cuda()
            label = data[1].cuda()
            # forward
            out = net(im)
            out = F.log_softmax(out, dim=1) # (b, n, h, w)
            loss = criterion(out, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            label_pred = out.max(dim=1)[1].data.cpu().numpy()
            label_true = label.data.cpu().numpy()
            for lbt, lbp in zip(label_true, label_pred):
                acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, 5)
                train_acc += acc
                train_acc_cls += acc_cls
                train_mean_iu += mean_iu
                train_fwavacc += fwavacc
            
        net = net.eval()
        eval_loss = 0
        eval_acc = 0
        eval_acc_cls = 0
        eval_mean_iu = 0
        eval_fwavacc = 0
        
        with torch.no_grad():
            for data in val_data:
                im = data[0].cuda()
                label = data[1].cuda()
                # forward
                out = net(im)
                out = F.log_softmax(out, dim=1)
                loss = criterion(out, label)
                eval_loss += loss.item()
                
                label_pred = out.max(dim=1)[1].data.cpu().numpy()
                label_true = label.data.cpu().numpy()
                for lbt, lbp in zip(label_true, label_pred):
                    acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, 5)
                    eval_acc += acc
                    eval_acc_cls += acc_cls
                    eval_mean_iu += mean_iu
                    eval_fwavacc += fwavacc
            
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        epoch_str = ('Epoch: {}, Train Loss: {:.5f}, Train Acc: {:.5f}, Train Mean IU: {:.5f},   Val Loss: {:.5f}, Val Acc: {:.5f}, Val Mean IU: {:.5f} '.format(
            e+1, train_loss / len(train_data), train_acc / len(train_dataset), train_mean_iu / len(train_dataset),
            eval_loss / len(val_data), eval_acc / len(val_dataset), eval_mean_iu / len(val_dataset)))
        time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
        print(epoch_str + time_str) # + ' lr: {}'.format(optimizer.learning_rate)
        print('--------------------------------------------------------------------------------------------------------------------------------------------------------')
        torch.save(net.state_dict(), './saved_model/deeplabv3/' + str(e+1) + '.pkl')            
    

