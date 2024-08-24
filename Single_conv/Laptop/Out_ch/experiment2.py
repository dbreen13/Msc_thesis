#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:33:47 2024

@author: dbreen
"""

#%%Initialize
import torch
import numpy as np
import tltorch as tl
import pickle
import torch.nn as nn
import tltorch
from tqdm import tqdm
from torch.autograd import Variable
import torch.optim as optim
import logging
from time import time, perf_counter
import time as timers
from datetime import datetime

logging.basicConfig(level = logging.INFO)

device = torch.device('cpu')
    

logger = logging.getLogger('Outch')
logger.setLevel(logging.INFO)

# Check if the logger already has a FileHandler
if not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
    fh = logging.FileHandler('Outch_nograd.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    
#%%CNN, consisting of one layer

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


def factorize_model(model, rank, factorization, decompose_weights, decomposition_kwargs, fixed_rank_modes):
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
    m=cnn_dict['n_epochs']
    lr=cnn_dict['lr']
    img_h=cnn_dict['img_h']
    img_w=cnn_dict['img_w']
    stride=cnn_dict['stride']
    padding=cnn_dict['padding']

    
    #params fact
    decompose_weights=True
    decompose=fact_dict['decompose']
    factorization=fact_dict['factorization']
    rank=fact_dict['rank']
    

    decomposition_kwargs = {'init': 'random'} if factorization == 'cp' else {}
    fixed_rank_modes = 'spatial' if factorization == 'tucker' else None
    ind=fact_dict['index']
    
    model=SimpleNet(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, num_classes=num_classes)
    
    if decompose==True:
        factorize_model(model, rank=rank,factorization=factorization, decomposition_kwargs=decomposition_kwargs, fixed_rank_modes=fixed_rank_modes, decompose_weights=decompose_weights)
        model.to(device)

    
    optimizer=optim.SGD(model.parameters(), lr=lr, weight_decay = 0.005, momentum = 0.9)

    model.eval()
    timers.sleep(60)
    now=datetime.now()
    sec_wait=60-now.second
    timers.sleep(sec_wait)
    start_training = perf_counter()
    if decompose==True:
        logger.info(f"dec-start-outch{out_channels}-inch{in_channels}-fact{factorization}-r{rank}-wh{img_w}-ind{ind}s")
    else:
        logger.info(f"bas-start-outch{out_channels}-inch{in_channels}-wh{img_w}-ind{ind}s")
    with torch.no_grad():
	    for _ in tqdm(range(m), desc="Forward Iterations"):

               output = model(Variable(x))

    if decompose==True:
        logger.info(f"dec-end-outch{out_channels}-inch{in_channels}-fact{factorization}-r{rank}-wh{img_w}-ind{ind}s")
    else:
        logger.info(f"bas-end-outch{out_channels}-inch{in_channels}-wh{img_w}-ind{ind}s")
    end_training = perf_counter()
    training_time = start_training - end_training
    print(training_time)
    return(model)

#%%Params CNN and Decomposition

#already defined params, steady for this type of experiment
img_h, img_w=[4,4]
kernel=3
padding=1
stride=2
in_ch=192
batch=128
num_classes=10
n_epochs=90000
lr=1e-5

cnn_dict={"in_channels": in_ch,
          "kernel_size": kernel,
          "batch_size": batch,
          "num_classes": num_classes,
          "n_epochs": n_epochs,
          "lr":lr,
          "img_h": img_h,
          "img_w": img_w,
          "stride": stride,
          "padding": padding}

compression=[0.1,0.25,0.5,0.75,0.9]
methods=['cp','tucker','tt', 'nd']
decompose=True


#create loop with all values to be determined

            
for out_ch in [256,192]:
    cnn_dict.update({"out_channels": out_ch})
    with open(f'/home/dbreen/Documents/tddl/toy_problems/Data/inch{in_ch}-wh{img_h}.pkl','rb') as f:  
        x = pickle.load(f)

    x=x.float()
    for method in methods:
        for c in compression:
           if method=='nd':
                   fact_dict={"decompose":False, "factorization":'c', "rank":0}
                   for ind in [3]:
                       fact_dict.update({'index':ind})
                       model=run_model(x,cnn_dict,fact_dict)
           else:
                   fact_dict={"decompose":decompose,
		                "factorization": method,
		                "rank" : c}
                   for ind in [3]:
                        fact_dict.update({'index':ind})
                        model=run_model(x,cnn_dict,fact_dict)


            


