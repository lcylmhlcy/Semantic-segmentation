# -*- coding:utf-8 -*-
import os
from PIL import Image
      
 
def img_label(img_path):
    j = img_path.index('.png')
    substr2 = img_path[j:]
    substr1 = img_path[0:j]
    label_name = substr1 + '_class' + substr2
    color_name = substr1 + '_color' + substr2

    return label_name, color_name

def splitimage(img_name, new_img_path, new_label_path, patch_size):
    img = Image.open(img_name)
    label_name, label_color_name = img_label(img_name)
    label = Image.open(label_name)
    label_color = Image.open(label_color_name)
    
    w, h = img.size
    w_new = (w // patch_size) * patch_size
    h_new = (h // patch_size) * patch_size
    img_new = img.crop((0, 0, w_new, h_new))
    label_new = label.crop((0, 0, w_new, h_new))
    label_color_new = label_color.crop((0, 0, w_new, h_new))
    
    s_img = os.path.split(img_name)
    fn_img = s_img[1].split('.')
    basename_img = fn_img[0]

    num = 1
    rowheight = patch_size
    colwidth = patch_size
    for r in range(h_new // patch_size):
        for c in range(w_new // patch_size):
            box = (c * colwidth, r * rowheight, (c + 1) * colwidth, (r + 1) * rowheight)
            
            img_new_path = os.path.join(new_img_path, basename_img + '_' + str(num) + '.png')
            img.crop(box).save(img_new_path)
            
            label_new_path = os.path.join(new_label_path, basename_img + '_' + str(num) + '.png')
            label.crop(box).save(label_new_path)
            
            # label_color_new_path = os.path.join(new_path, basename_img + '_' + str(num) + '_color.png')
            # label_color.crop(box).save(label_color_new_path)
            
            print(img_new_path)
            num = num + 1
    print('--------------------------------------------------------------------------------')

    
if __name__=='__main__':

    path = '/home/computer/lcy/pytorch/MyProject/Semantic-segmentation/dataset/CCFdataset/train/'  
    new_img_path = '/home/computer/lcy/pytorch/MyProject/Semantic-segmentation/data_process/new_dataset/image'
    new_label_path = '/home/computer/lcy/pytorch/MyProject/Semantic-segmentation/data_process/new_dataset/label'
    
    filename = os.listdir(path)
    for name in filename:
        if(name.find('_class') >=  0):
            continue
        elif(name.find('_color') >= 0):
            continue
        else:
            temp_img_path = os.path.join(path,name)
            splitimage(temp_img_path, new_img_path, new_label_path, 224)