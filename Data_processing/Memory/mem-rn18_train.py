#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:28:49 2024

@author: dbreen
"""
import ast

import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import tltorch
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.profiler import profile, record_function, ProfilerActivity

#data=pd.read_pickle('ranks_dataset.pkl')
#data=data.rename(columns={"Method":"Method",  "Label":"Rank"})

batch=128
in_features=[(batch,64,16,16),(batch,128,8,8), (batch,256,4,4), (batch,512,2,2)]
out_features=[(batch,64,16,16),(batch,128,8,8), (batch,256,4,4), (batch,512,2,2)]

layers=[63,60,57,54,51,47,41,38,35,28,25,22,19,15,6]    
compression=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
device = torch.device('cpu')

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

    
    
#%%MACS for decompsing
m=1
n=6
mem_dict={}
df = pd.DataFrame()
for method in ['cp', 'tucker', 'tt']:   
    for layer in layers: 
        for comp in compression:      
            stride=1
            padding=1
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
                padding=0
            else: 
                kernel=3
            with open(f'C:/Users/demib/Documents/Thesis/Memory/toy_problems/Data/inch{in_feature[1]}-wh{in_feature[2]}.pkl','rb') as f:  
                x = pickle.load(f)
            decomposition_kwargs = {'init': 'random'} if method == 'cp' else {}
            fixed_rank_modes = 'spatial' if method == 'tucker' else None
            decompose_weights=True
            model=SimpleNet(in_channels=in_feature[1], out_channels=out_feature[1], kernel_size=kernel, stride=stride, padding=padding, num_classes=10)
            factorize_model(model, rank=comp,factorization=method, decomposition_kwargs=decomposition_kwargs, fixed_rank_modes=fixed_rank_modes, decompose_weights=decompose_weights)
            model.to(device)
            with profile(activities=[ProfilerActivity.CPU], record_shapes=False, profile_memory=True) as prof:
                for _ in tqdm(range(m), desc="Forward Iterations"):
                    output = model(Variable(x))
            key_averages = prof.key_averages()
            mem = sum([item.cpu_memory_usage for item in key_averages])
            # if method=='tucker':
            #     rank=data.loc[data['Method']==f'dec-tucker-r{comp}-lay[{layer}]']
            #     rank=rank['Rank'].to_numpy()
            #     rank=ast.literal_eval(rank[0])
            #     rank=np.array(rank)
            #     macs=batch*(in_feature[2]**2*in_feature[1]*rank[1]+kernel*(np.prod(rank)+np.prod(rank[0:2])*rank[3]*kernel)+rank[0]*rank[1]*(in_feature[2]/stride)**2*kernel**2+rank[0]*out_feature[1]*(in_feature[2]/stride)**2)
            # elif method=='tt':
            #     rank=data.loc[data['Method']==f'dec-tt-r{comp}-lay[{layer}]']
            #     rank=rank['Rank'].to_numpy()
            #     rank=ast.literal_eval(rank[0])
            #     rank=np.array(rank)
            #     macs=batch*(in_feature[2]**2*in_feature[1]*rank[1]+rank[2]*(rank[1]*kernel*in_feature[2]/2*in_feature[2]+kernel*rank[3]*(in_feature[2]/2)**2)+(in_feature[2]/2)**2*out_feature[1]*rank[3])
            # elif method=='cp':
            #     rank=data.loc[data['Method']==f'dec-cp-r{comp}-lay[{layer}]']
            #     rank=rank['Rank'].to_numpy()
            #     rank=int(rank[0])
            #     macs=batch*(in_feature[2]**2*rank*in_feature[1]+rank*kernel*in_feature[2]/2*in_feature[2]+rank*kernel*(in_feature[2]/2)**2+(in_feature[2]/2)**2*out_feature[1]*rank)
            #     print(macs)
            # string=f'dec-{method}-r{comp}-lay[{layer}]'
            # df1 = pd.DataFrame({'Method': string, 'Rank':[np.array(rank)], 'Layer':layer,'MAC': macs, 'Comp':comp, 'Stride':stride, 'Kernel':kernel, 'In_ch':in_feature[1], 'Out_ch':out_feature[1]}, index=[0])
            # df=pd.concat([df,df1], ignore_index=True)
            key=f'outch{out_feature[1]}-inch{in_feature[1]}-fact{method}-r{comp}-wh{in_feature[2]}'
            mem_dict[key]={'Mem': np.round(mem*n,decimals=3)}            
df = pd.DataFrame.from_dict(mem_dict, orient='index', columns=['Mem'])  

save_path = "memrn18_inf.pkl"
with open(save_path, 'wb') as f:
    pickle.dump(df, f)          
# #%%MACS before decomposing      

# df_original = pd.DataFrame()
# for method in ['cp', 'tucker', 'tt']:   
#     for layer in layers: 
#         for comp in compression: 
#             if layer==6 or layer==15 or layer==9 or layer==25 or layer==19:
#                 in_feature=in_features[0]
#                 if layer==19 or layer==25:
#                     out_feature=out_features[1]
#                 else:
#                     out_feature=out_features[0]
#             elif layer==22 or layer==28 or layer==35 or layer==41:
#                 in_feature=in_features[1]
#                 if layer==28 or layer==22:
#                     out_feature=out_features[1]
#                 else:
#                     out_feature=out_features[2]
#             elif layer==38 or layer==47 or layer==51 or layer==57:
#                 in_feature=in_features[2]
#                 if layer==38 or layer==47:
#                     out_feature=out_features[2]
#                 else:
#                     out_feature=out_features[3]
#             else:
#                 in_feature=in_features[3]
#                 out_feature=out_features[3]
#             if layer==25 or layer==41 or layer==57:
#                 kernel=1
#             else: 
#                 kernel=3   
        
#             macs=in_feature[1]*out_feature[1]*kernel**2*out_feature[3]**2*batch
#             string=f'dec-{method}-r{comp}-lay[{layer}]'
#             df1 = pd.DataFrame({'Method': string, 'MAC_original': macs, 'In_feat':[np.array(in_feature)], 'Out_feat': [np.array(out_feature)], 'Dec': method}, index=[0])
#             df_original=pd.concat([df_original,df1], ignore_index=True)
            
# #%%Combine dataframes

# dataset_final=pd.concat([df['MAC'], df['Rank'], df['Comp'],df['Layer'],df_original['MAC_original'],df_original['In_feat'],df_original['Dec'],df_original['Out_feat'], df['Out_ch'], df['In_ch'], df['Stride'], df['Kernel']], axis=1)

