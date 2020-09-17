



#-------------------------------------
# Project: Few-shot for RCN
# Date: 2020.2.21
# Author: Zhiyu Xue
# All Rights Reserved
#-------------------------------------

import sys
import os
sys.path.append(os.getcwd())
from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import task_generator as tg
import os
import math
import argparse
import scipy as sp
import scipy.stats
from models import *
from utils import *

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 5) # n_way
parser.add_argument("-s","--sample_num_per_class",type = int, default = 5) # k_shot
parser.add_argument("-b","--batch_num_per_class",type = int, default = 10)
parser.add_argument("-e","--episode",type = int, default= 400000)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
parser.add_argument("-pv","--the_value_of_positive",type=float,default=1.0)
parser.add_argument("-nv","--the_value_of_negetive",type=float,default=-0.5)
parser.add_argument("-ft","--finetune_per_episode",type=int,default=200)
args = parser.parse_args()

def weights_init_kaiming(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv2d') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)

# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
PV = args.the_value_of_positive
NV = args.the_value_of_negetive
FT_EPISODE = args.finetune_per_episode
WIDTH = 19
NEW_WIDTH = 5

# name = "N_" + str(CLASS_NUM) + "_S_" + str(SAMPLE_NUM_PER_CLASS) + "_PV_" + str(PV) + "_NV_"  + str(NV) + "_l_" + str(LEARNING_RATE) 
name = "Conv4_argue5_meta_pair_N_" + str(CLASS_NUM) + "_S_" + str(SAMPLE_NUM_PER_CLASS) + "_B_" + str(BATCH_NUM_PER_CLASS) + "_l_" + str(LEARNING_RATE) 

def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatrain_folders,metaval_folders,metatest_folders = tg.mini_imagenet_folders()
    
    # Step 2: init neural networks and results dir
    print("init neural networks")

    path = makedir(name) # the path to save the results

    feature_encoder = FeatureEncoder_Conv4()
    middle_layer = Middle_Moudle_v3_ablation_meta(n_way=CLASS_NUM,k_shot=SAMPLE_NUM_PER_CLASS,width=WIDTH,batch_size=BATCH_NUM_PER_CLASS,new_width=NEW_WIDTH,in_channels=64)
    # explain_network = ExplainModule_v6_meta_pair(n_way=CLASS_NUM,k_shot=SAMPLE_NUM_PER_CLASS,width=NEW_WIDTH,batch_size=BATCH_NUM_PER_CLASS)
    explain_network = ExplainModule_v6_meta_pair_Conv4(n_way=CLASS_NUM,k_shot=SAMPLE_NUM_PER_CLASS,width=NEW_WIDTH,batch_size=BATCH_NUM_PER_CLASS)

    feature_encoder.apply(weights_init_kaiming)
    explain_network.apply(weights_init_kaiming)
    # feature_encoder.apply(weights_init_kaiming)
    # explain_network.generate_weights(neg_value=NV,pos_value=PV)

    feature_encoder.cuda(GPU)
    explain_network.cuda(GPU)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = ReduceLROnPlateau(feature_encoder_optim,mode="max",factor=0.5,patience=2,verbose=True)
    explain_network_optim = torch.optim.Adam(explain_network.parameters(),lr=LEARNING_RATE)
    explain_network_scheduler  = ReduceLROnPlateau(explain_network_optim,mode="max",factor=0.5,patience=2,verbose=True)

    total_accuracy = 0.0
    last_accuracy = 0.0
    # Create some lists to save data
    loss_list = []
    acc_list = []
    episode_list = []
    h_list = []
    episode_list = []
    

    for episode in range(EPISODE):
        explain_network.batch_size = BATCH_NUM_PER_CLASS
        middle_layer.batch_size = BATCH_NUM_PER_CLASS

        # init dataset
        # sample_dataloader is to obtain previous samples for compare
        # batch_dataloader is to batch samples for training
        task = tg.MiniImagenetTask(metatrain_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
        sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
        batch_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True,train_query_argue=True)

        # sample datas
        samples,sample_labels = sample_dataloader.__iter__().next() #(n*s)*3*84*84
        batches,batch_labels = batch_dataloader.__iter__().next()


        # calculate features
        sample_features = feature_encoder(Variable(samples).cuda(GPU)) # (n*s)*64*19*19
        # sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,19,19)
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)  # [n*b,n*s,64,19,19]


        batch_features = feature_encoder(Variable(batches).cuda(GPU)) # (n*b)*64*19*19
        # print(sample_labels)
        # print(batch_labels)
        # batch_features = batch_features.view(CLASS_NUM,BATCH_NUM_PER_CLASS,FEATURE_DIM,19,19) # [n,b,64,19,19]
        # batch_features = torch.transpose(batch_features,0,1) # [b,n,64,19,19]
        batch_features_ext = batch_features.unsqueeze(1).repeat(1,CLASS_NUM*SAMPLE_NUM_PER_CLASS,1,1,1) # [n*b,n*s,64,19,19]
        # print(sample_features_ext.size())
        # print(batch_features_ext.size())
        meta_input,distance_feature = middle_layer.forward(support_x=sample_features_ext,query_x=batch_features_ext) # [n*b,n*s,19*19]
        # print(distance_feature.size())
        # distance_feature = distance_feature.view([CLASS_NUM*BATCH_NUM_PER_CLASS,CLASS_NUM*SAMPLE_NUM_PER_CLASS*9*9])
        pre = explain_network(distance_feature,meta_input) # [n*b,n]
        # print(pre.size())
        
        mse = nn.MSELoss().cuda(GPU)
        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1).cuda(GPU))
        loss = mse(pre,one_hot_labels)
        # print(pre[0])
        # print(loss)
        # print(explain_network.layer.weight.data)
        feature_encoder_optim.zero_grad()
        explain_network_optim.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(),0.5)
        torch.nn.utils.clip_grad_norm( explain_network.parameters(),0.5)
        feature_encoder_optim.step()
        explain_network_optim.step()
        """
        if (episode+1)%FT_EPISODE == 0:
            print("fine tune")
            explain_network_optim.zero_grad()
            explain_network_optim.step()
            print(explain_network.layer.weight.data[0])
        """
        if (episode+1)%50 == 0:
            print("episode:",episode+1,"loss",loss.item())

        if (episode+1)%5000 == 0:
            print("val")
            val_accuracies = []
            with torch.no_grad():
                for i in range(TEST_EPISODE):
                    total_rewards = 0
                    task = tg.MiniImagenetTask(metaval_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,15)
                    sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
                    num_per_class = 5
                    explain_network.batch_size = num_per_class
                    middle_layer.batch_size = num_per_class
                    val_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=num_per_class,split="test",shuffle=False)
                    sample_images,sample_labels = sample_dataloader.__iter__().next()
                    for val_images,val_labels in val_dataloader:
                        val_images,val_labels = val_images.cuda(GPU),val_labels.cuda(GPU)
                        batch_size = val_labels.shape[0]
                        sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) # (n*s)*64*19*19
                        sample_features_ext = sample_features.unsqueeze(0).repeat(num_per_class*CLASS_NUM,1,1,1,1)  # [n*b,n*s,64,19,19]
                        val_features = feature_encoder(Variable(val_images).cuda(GPU)) # (n*b)*64*19*19
                        val_features_ext = val_features.unsqueeze(1).repeat(1,CLASS_NUM*SAMPLE_NUM_PER_CLASS,1,1,1) # [n*b,n*s,64,19,19]
                        meta_input,distance_feature = middle_layer.forward(support_x=sample_features_ext,query_x=val_features_ext) # [n*b,n*s,19*19]
                        # distance_feature = distance_feature.view([CLASS_NUM*num_per_class,CLASS_NUM*SAMPLE_NUM_PER_CLASS*9*9])
                        pre = explain_network(distance_feature,meta_input) # [n*b,n]
                        # print(pre[0])
                        # print(test_labels[0])

                        _,predict_labels = torch.max(pre.data,1)
                        rewards = [1 if predict_labels[j]==val_labels[j] else 0 for j in range(batch_size)]
                        
                        total_rewards += np.sum(rewards)

                    val_accuracy = total_rewards/1.0/CLASS_NUM/15
                    val_accuracies.append(val_accuracy)

                accuracy_val,h = mean_confidence_interval(val_accuracies)
                #迭代
                feature_encoder_scheduler.step(accuracy_val)
                explain_network_scheduler.step(accuracy_val)

        if episode%500 == 0:
            # test
            print("Testing...")
            accuracies = []
            with torch.no_grad():
                for i in range(TEST_EPISODE):
                    total_rewards = 0
                    task = tg.MiniImagenetTask(metatest_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,15)
                    sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
                    num_per_class = 5
                    explain_network.batch_size = num_per_class
                    middle_layer.batch_size = num_per_class
                    test_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=num_per_class,split="test",shuffle=False)

                    sample_images,sample_labels = sample_dataloader.__iter__().next()
                    for test_images,test_labels in test_dataloader:
                        test_images,test_labels = test_images.cuda(GPU),test_labels.cuda(GPU)
                        batch_size = test_labels.shape[0]
                        sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) # (n*s)*64*19*19
                        sample_features_ext = sample_features.unsqueeze(0).repeat(num_per_class*CLASS_NUM,1,1,1,1)  # [n*b,n*s,64,19,19]
                        test_features = feature_encoder(Variable(test_images).cuda(GPU)) # (n*b)*64*19*19
                        test_features_ext = test_features.unsqueeze(1).repeat(1,CLASS_NUM*SAMPLE_NUM_PER_CLASS,1,1,1) # [n*b,n*s,64,19,19]
                        meta_input,distance_feature = middle_layer.forward(support_x=sample_features_ext,query_x=test_features_ext) # [n*b,n*s,19*19]
                        # distance_feature = distance_feature.view([CLASS_NUM*num_per_class,CLASS_NUM*SAMPLE_NUM_PER_CLASS*9*9])
                        pre = explain_network(distance_feature,meta_input) # [n*b,n]
                        # print(pre[0])
                        # print(test_labels[0])

                        _,predict_labels = torch.max(pre.data,1)
                        rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(batch_size)]
                        
                        total_rewards += np.sum(rewards)

                    accuracy = total_rewards/1.0/CLASS_NUM/15
                    accuracies.append(accuracy)

                test_accuracy,h = mean_confidence_interval(accuracies)
                loss_list,acc_list,h_list = save_data(loss_list,acc_list,h_list,loss,test_accuracy,h)
                episode_list.append(episode)
                print("test accuracy:",test_accuracy,"h:",h)
                if test_accuracy > last_accuracy:
                    torch.save(feature_encoder.state_dict(),path+"feature_encoder.pkl")
                    torch.save(explain_network.state_dict(),path+"explain_network.pkl")

            np.savetxt(path+"acc.txt",acc_list)
            np.savetxt(path+"eposide.txt",episode_list)   
            np.savetxt(path+"h.txt",h_list)
            np.savetxt(path+"loss.txt",loss_list)







if __name__ == '__main__':
    main()
