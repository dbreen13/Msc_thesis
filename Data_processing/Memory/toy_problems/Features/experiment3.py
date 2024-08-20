#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:33:47 2024

@author: dbreen
"""

#%%Initialize
import torch
import pandas as pd
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
import tracemalloc
import os
from memory_profiler import profile
import pandas as pd
from torch.profiler import profile, record_function, ProfilerActivity

logging.basicConfig(level = logging.INFO)

device = torch.device('cpu')


logger = logging.getLogger('Feat')
logger.setLevel(logging.INFO)

# Check if the logger already has a FileHandler
if not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
    fh = logging.FileHandler('Feat.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

os.environ["KINETO_LOG_LEVEL"] = "5"     
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

    model.train()
    #timers.sleep(60)
    now=datetime.now()
    sec_wait=60-now.second
    #timers.sleep(sec_wait)
    start_training = perf_counter()
    # if decompose==True:
    #     logger.info(f"dec-start-outch{out_channels}-inch320-fact{factorization}-r{rank}-wh{img_w}-ind{ind}s")
    # else:
    #     logger.info(f"bas-start-outch{out_channels}-inch{in_channels}-wh{img_w}-ind{ind}s")
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU], record_shapes=False, profile_memory=True) as prof:
            for _ in tqdm(range(m), desc="Forward Iterations"):
                
                output = model(Variable(x))
                
        
                # batch_size, num_channels, height, width = output.size()
        
                # criterion = nn.CrossEntropyLoss()
                # labels = torch.randint(low=0, high=num_classes, size=(batch_size,), dtype=torch.long)
        
                # # Reshape labels to have the same spatial dimensions as the output tensor
                # labels = labels.view(batch_size, 1, 1).expand(batch_size, height, width)
                # optimizer.zero_grad()
                
                # # Compute the loss directly on reshaped output
                # loss = criterion(output, Variable(labels))
                
                # # Backward pass
                # loss.backward()
                # optimizer.step() 
            # if decompose==True:
            #     logger.info(f"dec-end-outch{out_channels}-inch320-fact{factorization}-r{rank}-wh{img_w}-ind{ind}s")
            # else:
            #     logger.info(f"bas-end-outch{out_channels}-inch{in_channels}-wh{img_w}-ind{ind}s")
    end_training = perf_counter()
    training_time = start_training - end_training
    print(training_time)
    key_averages = prof.key_averages()
    peak_memory = sum([item.cpu_memory_usage for item in key_averages])
    total_inclusive_memory = sum([item.cpu_memory_usage for item in key_averages])

    return(model,peak_memory / (1024 * 1024))
#%%Params CNN and Decomposition

#already defined params, steady for this type of experiment
# kernel=3
# padding=1
# stride=2
# in_chan=448
# out_chan=512
# batch=128
# num_classes=10
# n_epochs=1
# lr=1e-5
# n=1

# cnn_dict={"in_channels": in_chan,
#           "kernel_size": kernel,
#           "batch_size": batch,
#           "num_classes": num_classes,
#           "n_epochs": n_epochs,
#           "lr":lr,
#           "out_channels":out_chan,
#           "stride": stride,
#           "padding": padding}

# compression=[0.1,0.25,0.5,0.75,0.9]
# methods=['nd','tucker','tt', 'cp']
# decompose=True

# #create loop with all values to be determined
# mem_dict={}
                    
# for img_h in [2,4,6,8]:
#     img_w=img_h
#     cnn_dict.update({"img_h": img_h})
#     cnn_dict.update({"img_w":img_h})
#     with open(f'C:/Users/demib/Documents/Thesis/Memory/toy_problems/Data/inch{in_chan}-wh{img_h}.pkl','rb') as f:  
#         x = pickle.load(f)

#     x=x.float()
#     for method in methods:
#         if method=='nd':
#             fact_dict={"decompose":False, "factorization":'c', "rank":0}
#             for ind in [1]:
#                 fact_dict.update({'index':ind})
#                 model, mem=run_model(x,cnn_dict,fact_dict)
#                 key=f'bas-outch{out_chan}-inch{in_chan}-wh{img_w}'
#                 mem_dict[key]={'Mem': np.round(mem*n,decimals=3)}

#         else:
#             for c in compression:
#                 fact_dict={"decompose":decompose,
#                             "factorization": method,
#                             "rank" : c}
#                 for ind in [1]:
#                     fact_dict.update({'index':ind})
#                     model, mem=run_model(x,cnn_dict,fact_dict)
#                     key=f'outch{out_chan}-inch{in_chan}-fact{method}-r{c}-wh{img_w}'
#                     mem_dict[key]={'Mem': np.round(mem*n,decimals=3)}

            
# df = pd.DataFrame.from_dict(mem_dict, orient='index', columns=['Mem'])

# save_path = "mem_feat.pkl"
# with open(save_path, 'wb') as f:
#     pickle.dump(df, f)


#already defined params, steady for this type of experiment
img_w=4
img_h=4
padding=1
stride=2
in_chan=384
out_chan=512
batch=128
num_classes=10
n_epochs=1
n=1
lr=1e-5

cnn_dict={"in_channels": in_chan,
          "img_h":img_h,
          "img_w":img_w,
          "batch_size": batch,
          "num_classes": num_classes,
          "n_epochs": n_epochs,
          "lr":lr,
          "out_channels":out_chan,
          "stride": stride,
          "padding": padding}

compression=[0.1]
methods=['tt']
decompose=True

#create loop with all values to be determined

                 
mem_dict={}                 
for kernel in [1]:
    cnn_dict.update({"kernel_size":kernel})
    with open(f'C:/Users/demib/Documents/Thesis/Memory/toy_problems/Data/inch{in_chan}-wh{img_h}.pkl','rb') as f:  
        x = pickle.load(f)
    x=x.float()
    for method in methods:
        if method=='nd':
            fact_dict={"decompose":False, "factorization":'c', "rank":0}
            for ind in [1]:
                fact_dict.update({'index':ind})
                model, mem=run_model(x,cnn_dict,fact_dict)
                key=f'bas-outch{out_chan}-inch{in_chan}-kern{kernel}'
                mem_dict[key]={'Mem': np.round(mem*n,decimals=3)}
        else:
            for c in compression:
                fact_dict={"decompose":decompose,
                            "factorization": method,
                            "rank" : c}
                for ind in [1,2,3]:
                    fact_dict.update({'index':ind})
                    print(f'outch{out_chan}-inch{in_chan}-fact{method}-r{c}-wh{img_w}-kern{kernel}')
                    model, mem=run_model(x,cnn_dict,fact_dict)
                    key=f'outch{out_chan}-inch{in_chan}-fact{method}-r{c}-wh{img_w}-kern{kernel}'
                    mem_dict[key]={'Mem': np.round(mem*n,decimals=3)}
            
df = pd.DataFrame.from_dict(mem_dict, orient='index', columns=['Mem'])

save_path = "mem_kern.pkl"
with open(save_path, 'wb') as f:
    pickle.dump(df, f)

