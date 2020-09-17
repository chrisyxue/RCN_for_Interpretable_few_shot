# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:58:04 2020

@author: zhiyu xue
"""

import csv
import random
import os
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader,Dataset

path = ""
def load_csv(path):
    class_img_dict = {}
    with open(path) as f_csv:
        f_train = csv.reader(f_csv, delimiter=',')
        for row in f_train:
            if f_train.line_num == 1:
                continue
            img_name,img_class = row
            img_name = os.path.join("C:/Users/del/Desktop/可解释性项目/new idea/datas/CUB",img_name)
            
            if img_class in class_img_dict:
                class_img_dict[img_class].append(img_name)
            else:
                class_img_dict[img_class]=[]
                class_img_dict[img_class].append(img_name)
    f_csv.close()
    return class_img_dict


def mini_imagenet_folders():
    train_csv = 'C:/Users/del/Desktop/可解释性项目/new idea/datas/CUB/train.csv'
    test_csv = 'C:/Users/del/Desktop/可解释性项目/new idea/datas/CUB/test.csv'
    val_csv = 'C:/Users/del/Desktop/可解释性项目/new idea/datas/CUB/val.csv'
    
    metatrain_dict = load_csv(train_csv)
    metaval_dict = load_csv(val_csv)
    metatest_dict = load_csv(test_csv)
    return metatrain_dict,metaval_dict,metatest_dict

class MiniImagenetTask(object):

    def __init__(self, character_dict, num_classes, train_num,test_num):

        self.character_dict = character_dict
        self.class_list = character_dict.keys()
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num

        class_folders = random.sample(self.class_list,self.num_classes)
        labels = np.array(range(len(class_folders)))


        self.train_roots = []
        self.test_roots = []
        self.train_labels = []
        self.test_labels = []
        count = 0
        for c in class_folders:
            samples_path = self.character_dict[c]
            samples_path = random.sample(samples_path,train_num+test_num)
            train_samples_path = samples_path[:train_num]
            test_samples_path = samples_path[train_num:train_num+test_num]
            train_label = [labels[count]]*train_num
            test_label = [labels[count]]*test_num
            
            self.train_roots += train_samples_path
            self.test_roots += test_samples_path
            self.train_labels += train_label
            self.test_labels += test_label
            print(self.train_roots)
            count += 1
    def get_class(self, sample):
        return os.path.join(*sample.split('/')[:-1])
    
class FewShotDataset(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class MiniImagenet(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(MiniImagenet, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label 
    
metatrain_dict,metaval_dict,metatest_dict = mini_imagenet_folders()
i = MiniImagenetTask(metatrain_dict,5,5,10)