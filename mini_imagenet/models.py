# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:14:31 2020

@author: zhiyu xue
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
from dropblock import *
from utils import *

class Conv4(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(Conv4, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.view(out.size(0),-1)
        return out  # 64

class Conv4_new(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(Conv4_new, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.LeakyReLU(0.2, True),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.LeakyReLU(0.2, True),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.LeakyReLU(0.2, True))
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.LeakyReLU(0.2, True))

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.view(out.size(0),-1)
        return out  # 64  

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        
        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class ResNet(nn.Module):

    def __init__(self, block, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        return x

class ResNet_Classifier(nn.Module):

    def __init__(self, block, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5):
        self.inplanes = 3
        super(ResNet_Classifier, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        self.classifier = nn.Sequential(
            nn.Linear(640,64,bias=False),
            nn.Softmax()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0),x.size(1),-1).mean(-1,keepdim=False)
        x = self.classifier(x)
        # x = x.view(x.size(0), -1)
        return x



def FeatureEncoder(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(BasicBlock, keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model


def FeatureEncoder_Drop(keep_prob=1.0, avg_pool=False,drop_rate=0.1, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(BasicBlock, keep_prob=keep_prob, avg_pool=avg_pool, drop_rate=drop_rate,**kwargs)
    return model


def FeatureEncoderPre(keep_prob=1.0, avg_pool=False, **kwargs):
    model = ResNet_Classifier(BasicBlock, keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model

def FeatureEncoder_Conv4():
    model = Conv4()
    return model

class Global_Max_Pooling(nn.Module):
    def __init__(self,n_way,k_shot,batch_size):
        super(Global_Max_Pooling,self).__init__()
        self.n_way = n_way
        self.k_shot = k_shot
        # self.in_features = in_features
        self.batch_size = batch_size
        self.global_max_pooling = nn.AdaptiveAvgPool2d(1)
    def forward(self,x):
        "The size of x is [n*b,n*s,19,19]"
        # x = x.view(self.n_way*self.batch_size*self.n_way*self.k_shot,19,19)
        out = self.global_max_pooling(x)
        return out # [n*b,n*s,1,1]



def weights_init_FE(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def weights_init_EN(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def weights_init_EM_v2(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)


class Middle_Moudle_v3(nn.Module):
    def __init__(self,width,new_width = 5):
        super(Middle_Moudle_v3,self).__init__()
        self.width = width
        self.new_width = new_width
        self.global_max_pooling = nn.AdaptiveMaxPool2d(1)
    def forward(self,support_x,query_x):
        "Note that the size of support_x and query_x must be [n*b,n*s,64,19,19]"
        count = 0
        for i in range(self.width):
            for j in range(self.width):
                column = support_x[:,:,:,i,j].unsqueeze(-1)
                columns = column.unsqueeze(-1).repeat(1,1,1,self.width,self.width) # [n*b,n*s,64,19,19]
                similarity = self.global_max_pooling(torch.cosine_similarity(query_x,columns,dim=-3)).squeeze() # [n*b,n*s]
                similarity = similarity.unsqueeze(-1) # [n*b,n*s,1]
                if count==0:
                    out = similarity
                else:
                    out = torch.cat([out,similarity],dim=-1)
                count = count+1
        "out -> [n*b,n*s,19*19]"      
        return out

class Ablation_linear(nn.Module):
    def __init__(self,n_way,k_shot,width,batch_size,ratio=2):
        super(Ablation_linear,self).__init__()
        self.batch_size = batch_size
        self.n_way = n_way
        self.k_shot = k_shot
        self.in_features = width*width
        self.linear = nn.Linear(self.in_features,1,bias=False)
        self.soft_max = nn.Softmax()
        # self.learner_2 = nn.Linear(self.k_shot,1,bias=False)
            
    def forward(self,x):
        "The size of x is [n*b,n*s,19*19]"
        x = self.linear(x)
        x = x.view([self.n_way*self.batch_size*self.n_way,self.k_shot]) # [n*b*n,s]
        if self.k_shot == 1:
            out = x.view([-1,self.n_way])
        else:
            out =  x.mean(-1).view([-1,self.n_way])
        out = self.soft_max(out)
        return out

class Middle_Moudle_v3_ablation_meta(nn.Module):
    def __init__(self,width,n_way,k_shot,batch_size,in_channels=640,new_width = 5):
        super(Middle_Moudle_v3_ablation_meta,self).__init__()
        self.in_channels = in_channels
        self.n_way = n_way
        self.k_shot = k_shot
        self.batch_size = batch_size
        self.width = width
        self.new_width = new_width
        self.global_max_pooling = nn.AdaptiveMaxPool2d(1)
        self.global_ave_pooling = nn.AdaptiveAvgPool2d([self.new_width,self.new_width])
    def forward(self,support_x,query_x):
        "Note that the size of support_x and query_x must be [n*b,n*s,64,19,19]"
        support_x,query_x = support_x.view([self.n_way*self.batch_size*self.n_way*self.k_shot,self.in_channels,self.width,self.width]),query_x.view([self.n_way*self.batch_size*self.n_way*self.k_shot,self.in_channels,self.width,self.width])
        if self.new_width == 1:
            support_x = self.global_ave_pooling(support_x)
            query_x = self.global_ave_pooling(query_x)
        elif self.new_width != self.width:
            support_x = self.global_ave_pooling(support_x)
            query_x = self.global_ave_pooling(query_x)
        else:
            pass

        support_x,query_x = support_x.view([self.n_way*self.batch_size,self.n_way*self.k_shot,self.in_channels,self.new_width,self.new_width]),query_x.view([self.n_way*self.batch_size,self.n_way*self.k_shot,self.in_channels,self.new_width,self.new_width])
        meta_input = torch.cat([support_x,query_x],dim=2)
        count = 0
        # print(support_x.size())
        for i in range(self.new_width):
            for j in range(self.new_width):
                column = support_x[:,:,:,i,j].unsqueeze(-1)
                columns = column.unsqueeze(-1).repeat(1,1,1,self.new_width,self.new_width) # [n*b,n*s,64,19,19]
                similarity = self.global_max_pooling(torch.cosine_similarity(query_x,columns,dim=-3)).squeeze() # [n*b,n*s]
                similarity = similarity.unsqueeze(-1) # [n*b,n*s,1]
                if count==0:
                    out = similarity
                else:
                    out = torch.cat([out,similarity],dim=-1)
                count = count+1
        "out -> [n*b,n*s,19*19]"      
        return meta_input,out

class ExplainModule_v6_meta_pair(nn.Module):
    def __init__(self,n_way,k_shot,width,batch_size,ratio=2):
        super(ExplainModule_v6_meta_pair,self).__init__()
        self.batch_size = batch_size
        self.n_way = n_way
        self.k_shot = k_shot
        self.width = width
        self.in_features = width*width
        self.in_channels = 640*2
        # self.layer = nn.Linear(k_shot,1,bias=False)
        self.soft_max = nn.Softmax()
        self.meta_learner = nn.Sequential(
            nn.Conv2d(self.in_channels,640,stride=1,kernel_size=1,bias=False),
            nn.BatchNorm2d(640),
            nn.ReLU(),
            nn.Conv2d(640,64,stride=1,kernel_size=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,1,stride=1,kernel_size=1,bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
            
    def forward(self,x,meta_input):
        "The size of x is [n*b,n*s,19*19]"
        "The size of meta_input is [n*b,n*s,channels,width,width]"
        meta_input = meta_input.view([self.n_way*self.batch_size*self.n_way*self.k_shot,self.in_channels,self.width,self.width])
        meta_weight = self.meta_learner(meta_input)
        meta_weight = meta_weight.view([self.n_way*self.batch_size,self.n_way*self.k_shot,self.in_features])
        x = (x*meta_weight).sum(-1) # [n*b,n*s]
        x = x.view([self.n_way*self.batch_size*self.n_way,self.k_shot]) # [n*b*n,s]
        out = torch.mean(x,dim=-1).view([-1,self.n_way])
        out = self.soft_max(out)
        return out


class ExplainModule_v6_meta_pair_Conv4(nn.Module):
    def __init__(self,n_way,k_shot,width,batch_size,ratio=2):
        super(ExplainModule_v6_meta_pair_Conv4,self).__init__()
        self.batch_size = batch_size
        self.n_way = n_way
        self.k_shot = k_shot
        self.width = width
        self.in_features = width*width
        self.in_channels = 64*2
        # self.layer = nn.Linear(k_shot,1,bias=False)
        self.soft_max = nn.Softmax()
        self.meta_learner = nn.Sequential(
            nn.Conv2d(self.in_channels,64,stride=1,kernel_size=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,1,stride=1,kernel_size=1,bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
            
    def forward(self,x,meta_input):
        "The size of x is [n*b,n*s,19*19]"
        "The size of meta_input is [n*b,n*s,channels,width,width]"
        meta_input = meta_input.view([self.n_way*self.batch_size*self.n_way*self.k_shot,self.in_channels,self.width,self.width])
        meta_weight = self.meta_learner(meta_input)
        meta_weight = meta_weight.view([self.n_way*self.batch_size,self.n_way*self.k_shot,self.in_features])
        x = (x*meta_weight).sum(-1) # [n*b,n*s]
        x = x.view([self.n_way*self.batch_size*self.n_way,self.k_shot]) # [n*b*n,s]
        out = torch.mean(x,dim=-1).view([-1,self.n_way])
        out = self.soft_max(out)
        return out


