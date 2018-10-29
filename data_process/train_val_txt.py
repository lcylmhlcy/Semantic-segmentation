# -*- coding:utf-8 -*-
import time     
import os  
import shutil
import random
 
 
def readFilename(path):
    allfile = []
    filelist = os.listdir(path)
 
    for filename in filelist:
        filepath = os.path.join(path, filename)
        allfile.append(filepath)
        
    return allfile
 
 
 
if __name__ == '__main__':
    path="/home/computer/lcy/pytorch/MyProject/Semantic-segmentation/data_process/new_dataset/image/"
    allfile=readFilename(path)
    random.shuffle(allfile)  
    
    allname=[]
    train_txtpath = "/home/computer/lcy/pytorch/MyProject/Semantic-segmentation/data_process/new_dataset/train.txt"
    val_txtpath = "/home/computer/lcy/pytorch/MyProject/Semantic-segmentation/data_process/new_dataset/val.txt"
    
    for i,name in enumerate(allfile):
             
        file_name = name.split("/")[-1].split(".")[0]
        print(file_name)
        
        if i < 2852:
            with open(train_txtpath,'a+') as tfp:
                tfp.write(file_name+"\n")
        else:
            with open(val_txtpath,'a+') as vfp:
                vfp.write(file_name+"\n")