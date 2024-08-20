#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:10:21 2024

@author: dbreen
"""
import torch
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity
from torchprofile import profile_macs
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
#from thop import profile

import sys,os
import tltorch
sys.path.append("/home/Documents/tddl")
from tddl.factorizations import factorize_network
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import torchvision.transforms as transforms
from time import time, perf_counter
import logging
logging.basicConfig(level = logging.INFO)
from datetime import datetime
import time as timers
import numpy as np


logger=logging.getLogger('Layertest_inf')
#create a fh
fh=logging.FileHandler('laytesten_inf.log')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

device = torch.device('cpu')
import pandas as pd

class SimpleNet(nn.Module):
    def __init__(self,
        out_channels:int,
        kernel_size:int,
        stride:int,
        padding:int,
        num_classes:int,
        in_channels:int,
            ):
        super(SimpleNet,self).__init__()
    
        self.conv1=nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        
    def forward(self,x):
        x=self.conv1(x)
        return x


def factorize_model(model, rank,factorization, decomposition_kwargs, fixed_rank_modes,decompose_weights):
    children=dict(model.named_children())
    for child in children.items():
        model._modules['conv1'] = tltorch.FactorizedConv.from_conv(
                    child[1], 
                    rank=rank, 
                    decompose_weights=decompose_weights, 
                    factorization=factorization,
                    decomposition_kwargs=decomposition_kwargs,
                    fixed_rank_modes=fixed_rank_modes,
                    implementation='factorized',
                )
        
def run_model(x,cnn_dict, fact_dict):
    #params cnn

    in_channels=cnn_dict['in_channels']
    out_channels=cnn_dict['out_channels']
    kernel_size=cnn_dict['kernel_size']
    batch_size=cnn_dict['batch_size']
    num_classes=cnn_dict['num_classes']
    n_epochs=cnn_dict['n_epochs']
    lr=cnn_dict['lr']
    img_h=cnn_dict['img_h']
    img_w=cnn_dict['img_w']
    m=cnn_dict['iterations']
    stride=cnn_dict['stride']

    
    #params fact
    decompose_weights=False
    td_init=0.02
    return_error=False
    decompose=fact_dict['decompose']
    layers=fact_dict['layers']
    factorization=fact_dict['factorization']
    rank=fact_dict['rank']
    decomposition_kwargs = {'init': 'random'} if factorization == 'cp' else {}
    fixed_rank_modes = 'spatial' if factorization == 'tucker' else None
    ind=fact_dict['index']
    
    model=SimpleNet(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=1, num_classes=num_classes)
    model.to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(), lr=lr, weight_decay = 0.005, momentum = 0.9)
    
    
        
    #start_time = time()

   
    if decompose==True:
        factorize_model(model, rank,factorization, decomposition_kwargs, fixed_rank_modes, decompose_weights)
        model.to(device)
    
    
    model.train()
    logger.info(f"start-forward-outch{out_channels}-inch{in_channels}-fact{factorization}-ind{ind}s")
    start_training = perf_counter()
    for _ in tqdm(range(m), desc="Forward Iterations"):
        output = model(x)
        
    logger.info(f"end-forward-outch{out_channels}-inch{in_channels}-fact{factorization}-ind{ind}s")
            #end_time = time()    
    end_training = perf_counter()
    training_time = start_training - end_training
    print(training_time)
    batch_size, num_channels, height, width = output.size()
 
    output_shape=model(x).shape    
    # Reshape output for CrossEntropyLoss
    batch_size, num_channels, height, width = output.size()
    reshaped_output = output.view(batch_size, num_channels * height * width)

    # Create random target labels (assuming 10 classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    # Define CrossEntropyLoss criterion
    criterion = nn.CrossEntropyLoss()
 
    list_out=[]
    list_labels=[]
    for i in range(2):
        output_new=torch.randn(output.size(),requires_grad=True).to(device)
        reshaped_output = output_new.view(batch_size, num_channels * height * width)
        list_out.append(reshaped_output)
        list_labels.append(torch.randint(0, num_classes, (batch_size,)))

    #elapsed_time = end_time - start_time

    #print(f"Time taken: {elapsed_time} seconds")

    #start_time = time()
    start_training = perf_counter()
    logger.info(f"start-back-outch{out_channels}-inch{in_channels}-fact{factorization}-ind{ind}s")
    for i in tqdm(range(m), desc="Backward Iterations"):
        if i % 2 == 0:
            output_it=list_out[0]
        else:
            output_it=list_out[1]
            
        optimizer.zero_grad()
        # Compute the loss directly on reshaped output
        loss = criterion(output_it, labels)
        # Backward pass
        loss.backward()
        optimizer.step() 
    logger.info(f"end-back-outch{out_channels}-inch{in_channels}-fact{factorization}-ind{ind}s")
    end_training = perf_counter()
    training_time = start_training - end_training
    print(training_time)
    return model
    
            
    

#%%



#Define cnn model parameter
in_channels=[64, 128, 256, 512]
out_channels=[64, 128, 256, 512]
kernel_size=1
num_classes=2
padding=1
stride=2
batch_size=128
n_epochs=25
lr=1e-5
it=2000

#parameters the dataset
input_size=in_channels
img_h=4
img_w=4
#Define factorization parameters
decompose=False
layers=[0]
factorizations=['cp','tucker','tt']
rank=0.1
ind=0

#model_trained, tab=run_model(cnn_dict, fact_dict)
batch=128
in_features=[(batch,64,16,16),(batch,128,8,8), (batch,256,4,4), (batch,512,2,2)]
out_features=[(batch,64,16,16),(batch,128,8,8), (batch,256,4,4), (batch,512,2,2)]
layers=[63,60,57,54,51,47,41,38,35,28,25,22,19,15,6]    
compression=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]




for method in ['cp']:
        for i in compression:
            for layer in layers:
                for j in range(2):
                    stride=1
                    if layer==6 or layer==15 or layer==9 or layer==25 or layer==19:
                        in_feature=in_features[0]
                        if layer==19 or layer==25:
                            out_feature=out_features[1]
                            stride=2
                        else:
                            out_feature=out_features[0]
                    elif layer==22 or layer==28 or layer==35 or layer==41:
                        in_feature=in_features[1]
                        if layer==28 or layer==22:
                            out_feature=out_features[1]
                        else:
                            out_feature=out_features[2]
                            stride=2
                    elif layer==38 or layer==47 or layer==51 or layer==57:
                        in_feature=in_features[2]
                        if layer==38 or layer==47:
                            out_feature=out_features[2]
                        else:
                            out_feature=out_features[3]
                            stride=2
                    else:
                        in_feature=in_features[3]
                        out_feature=out_features[3]
                    if layer==25 or layer==41 or layer==57:
                        kernel=1
                    else: 
                        kernel=3
                        
                    file_path=f'[128,{in_feature[1]},{in_feature[2]},{in_feature[2]}].pickle'
                    with open(file_path, 'rb') as file:
                        # Deserialize and retrieve the variable from the file
                        x = pickle.load(file)
                    #macs=profile_macs(model,(x,))
                    #macs, params = profile(model, inputs=(x, ))
                    x=x.repeat(128,1,1,1)
                    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                       #     with record_function("model_inference"):
                           
                    cnn_dict={"in_channels": in_feature[1],
                                  "out_channels": out_feature[1],
                                  "kernel_size": kernel,
                                  "batch_size": batch_size,
                                  "num_classes": num_classes,
                                  "n_epochs": n_epochs,
                                  "lr":lr,
                                  "img_h": in_feature[2],
                                  "img_w": in_feature[2],
                                  "iterations": it,
                                  "stride": stride,
                                  }
                    
                    fact_dict={"decompose":decompose,
                                "layers": layers,
                                "factorization": method,
                                "rank" : i,
                                "index": ind,
                                }
            
                    model_trained=run_model(x,cnn_dict, fact_dict)




for method in ['tucker']:
        for i in compression:
            for layer in layers:
                for j in range(5):
                    stride=1
                    if layer==6 or layer==15 or layer==9 or layer==25 or layer==19:
                        in_feature=in_features[0]
                        if layer==19 or layer==25:
                            out_feature=out_features[1]
                            stride=2
                        else:
                            out_feature=out_features[0]
                    elif layer==22 or layer==28 or layer==35 or layer==41:
                        in_feature=in_features[1]
                        if layer==28 or layer==22:
                            out_feature=out_features[1]
                        else:
                            out_feature=out_features[2]
                            stride=2
                    elif layer==38 or layer==47 or layer==51 or layer==57:
                        in_feature=in_features[2]
                        if layer==38 or layer==47:
                            out_feature=out_features[2]
                        else:
                            out_feature=out_features[3]
                            stride=2
                    else:
                        in_feature=in_features[3]
                        out_feature=out_features[3]
                    if layer==25 or layer==41 or layer==57:
                        kernel=1
                    else: 
                        kernel=3
                
                    cnn_dict={"in_channels": in_feature[1],
                                  "out_channels": out_feature[1],
                                  "kernel_size": kernel,
                                  "batch_size": batch_size,
                                  "num_classes": num_classes,
                                  "n_epochs": n_epochs,
                                  "lr":lr,
                                  "img_h": in_feature[2],
                                  "img_w": in_feature[2],
                                  "iterations": it,
                                  "stride": stride,
                                  }
                    
                    fact_dict={"decompose":decompose,
                                "layers": layers,
                                "factorization": method,
                                "rank" : i,
                                "index": ind,
                                }
            
                    model_trained=run_model(cnn_dict, fact_dict)

                




for method in ['tt']:
        for i in compression:
            for layer in layers:
                for j in range(5):
                    stride=1
                    if layer==6 or layer==15 or layer==9 or layer==25 or layer==19:
                        in_feature=in_features[0]
                        if layer==19 or layer==25:
                            out_feature=out_features[1]
                            stride=2
                        else:
                            out_feature=out_features[0]
                    elif layer==22 or layer==28 or layer==35 or layer==41:
                        in_feature=in_features[1]
                        if layer==28 or layer==22:
                            out_feature=out_features[1]
                        else:
                            out_feature=out_features[2]
                            stride=2
                    elif layer==38 or layer==47 or layer==51 or layer==57:
                        in_feature=in_features[2]
                        if layer==38 or layer==47:
                            out_feature=out_features[2]
                        else:
                            out_feature=out_features[3]
                            stride=2
                    else:
                        in_feature=in_features[3]
                        out_feature=out_features[3]
                    if layer==25 or layer==41 or layer==57:
                        kernel=1
                    else: 
                        kernel=3
                
                    cnn_dict={"in_channels": in_feature[1],
                                  "out_channels": out_feature[1],
                                  "kernel_size": kernel,
                                  "batch_size": batch_size,
                                  "num_classes": num_classes,
                                  "n_epochs": n_epochs,
                                  "lr":lr,
                                  "img_h": in_feature[2],
                                  "img_w": in_feature[2],
                                  "iterations": it,
                                  "stride": stride,
                                  }
                    
                    fact_dict={"decompose":decompose,
                                "layers": layers,
                                "factorization": method,
                                "rank" : i,
                                "index": ind,
                                }
            
                    model_trained=run_model(cnn_dict, fact_dict)

