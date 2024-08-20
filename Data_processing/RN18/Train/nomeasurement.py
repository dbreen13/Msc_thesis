#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 13:53:11 2024

@author: dbreen
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.close('all')

data=pd.read_pickle('allinfo.pkl')

data_tt_tuck=pd.read_pickle('dataset_tt_tucker.pkl').transpose()
data_tt_tuck=data_tt_tuck[1:]

data_eng=pd.read_pickle('dataset_final.pkl').transpose()
baseline=data_eng.iloc[:1]
data_eng=data_eng[1:]


data.reset_index(drop=True, inplace=True)
data_eng.reset_index(drop=True, inplace=True)
data_tt_tuck.reset_index(drop=True,inplace=True)


datacp_tt_tuck=pd.concat([data_eng,data_tt_tuck], axis=0)

datacp_tt_tuck.reset_index(drop=True, inplace=True)
data=pd.concat([datacp_tt_tuck,data], axis=1)
# data.set_index(index)


maskcp=data['Dec'].isin(['cp'])

datacp=data[maskcp]




data['MAC_original']=data['MAC_original']*128
#%%cp decomposition
maskcp=data['Dec'].isin(['cp'])
datacp=data[maskcp]

data64=datacp[datacp['In_feat'].apply(lambda x: np.array_equal(x, np.array([128, 64, 16, 16])))]
data64=data64[data64['Out_feat'].apply(lambda x: np.array_equal(x, np.array([128, 128, 8, 8])))]


data128=datacp[datacp['In_feat'].apply(lambda x: np.array_equal(x, np.array([128, 128, 8, 8])))]
data128=data128[data128['Out_feat'].apply(lambda x: np.array_equal(x, np.array([128, 256, 4, 4])))]

data256=datacp[datacp['In_feat'].apply(lambda x: np.array_equal(x, np.array([128, 256, 4, 4])))]
data256=data256[data256['Out_feat'].apply(lambda x: np.array_equal(x, np.array([128, 512, 2, 2])))]

data512=datacp[datacp['In_feat'].apply(lambda x: np.array_equal(x, np.array([128, 512, 2, 2])))]
data512=data512[data512['Out_feat'].apply(lambda x: np.array_equal(x, np.array([128, 512, 2, 2])))]


plt.figure()
plt.scatter(data64['Comp'].to_numpy(), data64['MAC'].to_numpy(dtype=int), color='k')
plt.scatter(data128['Comp'].to_numpy(), data128['MAC'].to_numpy(dtype=int), color='b')
plt.scatter(data256['Comp'].to_numpy(), data256['MAC'].to_numpy(dtype=int), color='r')
plt.scatter(data512['Comp'].to_numpy(), data512['MAC'].to_numpy(dtype=int), color='g')
plt.xlabel('Ranks')
plt.ylabel('MACs')
plt.legend(['feat=16x16', 'feat=8x8', 'feat=4x4', 'feat=2x2'])


plt.figure()
plt.scatter(data64['Comp'].to_numpy(), data64['MAC']/data64['MAC_original'].to_numpy(dtype=int), color='k')
plt.scatter(data128['Comp'].to_numpy(), data128['MAC']/data128['MAC_original'].to_numpy(dtype=int), color='b')
plt.scatter(data256['Comp'].to_numpy(), data256['MAC']/data256['MAC_original'].to_numpy(dtype=int), color='r')
plt.scatter(data512['Comp'].to_numpy(), data512['MAC']/data512['MAC_original'].to_numpy(dtype=int), color='g')
plt.xlabel('Ranks')
plt.ylabel('MACs_decomp/MACs_notdecp')
plt.legend(['feat=16x16', 'feat=8x8', 'feat=4x4', 'feat=2x2'])

#%%tucker decomposition
masktuck=data['Dec'].isin(['tucker'])
datatuck=data[masktuck]

data64=datatuck[datatuck['In_feat'].apply(lambda x: np.array_equal(x, np.array([128, 64, 16, 16])))]
data64=data64[data64['Out_feat'].apply(lambda x: np.array_equal(x, np.array([128, 64, 16, 16])))]


data128=datatuck[datatuck['In_feat'].apply(lambda x: np.array_equal(x, np.array([128, 128, 8, 8])))]
data128=data128[data128['Out_feat'].apply(lambda x: np.array_equal(x, np.array([128, 128, 8, 8])))]

data256=datatuck[datatuck['In_feat'].apply(lambda x: np.array_equal(x, np.array([128, 256, 4, 4])))]
data256=data256[data256['Out_feat'].apply(lambda x: np.array_equal(x, np.array([128, 256, 4, 4])))]

data512=datatuck[datatuck['In_feat'].apply(lambda x: np.array_equal(x, np.array([128, 512, 2, 2])))]
data512=data512[data512['Out_feat'].apply(lambda x: np.array_equal(x, np.array([128, 512, 2, 2])))]
plt.figure()
plt.scatter((data64['Comp']).to_numpy(), data64['MAC'].to_numpy(dtype=int), color='k')
plt.scatter(data128['Comp'].to_numpy(), data128['MAC'].to_numpy(dtype=int), color='b')
plt.scatter(data256['Comp'].to_numpy(), data256['MAC'].to_numpy(dtype=int), color='r')
plt.scatter(data512['Comp'].to_numpy(), data512['MAC'].to_numpy(dtype=int), color='g')
plt.xlabel('Ranks')
plt.ylabel('MACs')
plt.legend(['feat=16x16', 'feat=8x8', 'feat=4x4', 'feat=2x2'])

#TT decomposition
masktt=data['Dec'].isin(['tt'])
datatt=data[masktt]

data64=datatt[datatt['In_feat'].apply(lambda x: np.array_equal(x, np.array([128, 64, 16, 16])))]
data64=data64[data64['Out_feat'].apply(lambda x: np.array_equal(x, np.array([128, 64, 16, 16])))]


data128=datatt[datatt['In_feat'].apply(lambda x: np.array_equal(x, np.array([128, 128, 8, 8])))]
data128=data128[data128['Out_feat'].apply(lambda x: np.array_equal(x, np.array([128, 128, 8, 8])))]

data256=datatt[datatt['In_feat'].apply(lambda x: np.array_equal(x, np.array([128, 256, 4, 4])))]
data256=data256[data256['Out_feat'].apply(lambda x: np.array_equal(x, np.array([128, 256, 4, 4])))]

data512=datatt[datatt['In_feat'].apply(lambda x: np.array_equal(x, np.array([128, 512, 2, 2])))]
data512=data512[data512['Out_feat'].apply(lambda x: np.array_equal(x, np.array([128, 512, 2, 2])))]
plt.figure()
plt.scatter((data64['Comp']).to_numpy(), data64['MAC'].to_numpy(dtype=int), color='k')
plt.scatter(data128['Comp'].to_numpy(), data128['MAC'].to_numpy(dtype=int), color='b')
plt.scatter(data256['Comp'].to_numpy(), data256['MAC'].to_numpy(dtype=int), color='r')
plt.scatter(data512['Comp'].to_numpy(), data512['MAC'].to_numpy(dtype=int), color='g')
plt.xlabel('Ranks')
plt.ylabel('MACs')
plt.legend(['feat=16x16', 'feat=8x8', 'feat=4x4', 'feat=2x2'])

#%%stride vs macs

for in_ch in [64,128,256]: 
        datastr_cp=datacp[datacp['Stride'].isin([2])]
        data_in=datastr_cp[datastr_cp['In_ch'].isin([in_ch])]
        data_kern2=data_in[data_in['Kernel'].isin([3])]
        
        MAC_str2=data_kern2['MAC'].to_numpy()
        
        
        datastr_cp=datacp[datacp['Stride'].isin([1])]
        data_in=datastr_cp[datastr_cp['In_ch'].isin([in_ch])]
        data_kern1=data_in[data_in['Kernel'].isin([3])]
        
        MAC_str1=data_kern1['MAC'].to_numpy()
        
        plt.figure()
        plt.scatter(data_kern2['Comp'].to_numpy(), MAC_str2,color='b')
        plt.scatter(data_kern1['Comp'].to_numpy(), MAC_str1)
        plt.legend(['Stride=1', 'Stride=2'])
 
        plt.title(f'Influence stride on MAC for in_ch={in_ch}')
        plt.xlabel('Compression')
        plt.ylabel('MACs')
        plt.savefig(f'Figures/Stride/stride_inch{in_ch}.png')
        
#%%outch vs macs

for stri in [1,2]: 
        datastr_cp=datacp[datacp['Stride'].isin([stri])]
        data_out=datastr_cp[datastr_cp['In_ch'].isin([64])]
        data_kern64=data_out[data_out['Kernel'].isin([3])]
        
        MAC_out64=data_kern64['MAC'].to_numpy()
        
        datastr_cp=datacp[datacp['Stride'].isin([stri])]
        data_out=datastr_cp[datastr_cp['In_ch'].isin([128])]
        data_kern128=data_out[data_out['Kernel'].isin([3])]
        
        MAC_out128=data_kern128['MAC'].to_numpy()
        
        datastr_cp=datacp[datacp['Stride'].isin([stri])]
        data_out=datastr_cp[datastr_cp['In_ch'].isin([256])]
        data_kern256=data_out[data_out['Kernel'].isin([3])]
        
        MAC_out256=data_kern256['MAC'].to_numpy()
        
        datastr_cp=datacp[datacp['Stride'].isin([stri])]
        data_out=datastr_cp[datastr_cp['In_ch'].isin([512])]
        data_kern512=data_out[data_out['Kernel'].isin([3])]
        
        MAC_out512=data_kern512['MAC'].to_numpy()
        
        plt.figure()
        plt.scatter(data_kern64['Comp'].to_numpy(), MAC_out64,color='b')
        plt.scatter(data_kern128['Comp'].to_numpy(), MAC_out128)
        plt.scatter(data_kern256['Comp'].to_numpy(), MAC_out256)
        if stri==1:
            plt.scatter(data_kern512['Comp'].to_numpy(), MAC_out512)
        plt.legend(['Params=16384', 'Params=8192', 'Params=4096', 'Params=2048'])

        plt.title(f'Influence number of input to MACs for stride={stri}')
        plt.xlabel('Compression')
        plt.ylabel('MACs')
        plt.savefig(f'Figures/Features/feat_str{stri}.png')

#%%Energy
datacp=datacp[datacp['In_feat'].apply(lambda x: np.array_equal(x, np.array([128, 256,4,4])))]
datacp=datacp[datacp['Out_feat'].apply(lambda x: np.array_equal(x, np.array([128, 256,4,4])))]


plt.figure()
plt.scatter( datacp['Comp'],datacp['energy(kWh)'].to_numpy())

#%%
maskcp=data['Dec'].isin(['cp'])
datacp=data[maskcp]

data64=datacp[datacp['In_feat'].apply(lambda x: np.array_equal(x, np.array([128, 64, 16, 16])))]
data64=data64[data64['Out_feat'].apply(lambda x: np.array_equal(x, np.array([128, 64, 16, 16])))]


data128=datacp[datacp['In_feat'].apply(lambda x: np.array_equal(x, np.array([128, 128, 8, 8])))]
data128=data128[data128['Out_feat'].apply(lambda x: np.array_equal(x, np.array([128, 128, 8, 8])))]

data256=datacp[datacp['In_feat'].apply(lambda x: np.array_equal(x, np.array([128, 256, 4, 4])))]
data256=data256[data256['Out_feat'].apply(lambda x: np.array_equal(x, np.array([128, 256, 4, 4])))]

data512=datacp[datacp['In_feat'].apply(lambda x: np.array_equal(x, np.array([128, 512, 2, 2])))]
data512=data512[data512['Out_feat'].apply(lambda x: np.array_equal(x, np.array([128, 512, 2, 2])))]

data512=data512[data512['Kernel'].isin([3])]


plt.figure()
plt.scatter(data64['Comp'].to_numpy(), data64['energy(kWh)'].to_numpy(), color='k')
plt.scatter(data128['Comp'].to_numpy(), data128['energy(kWh)'].to_numpy(), color='b')
plt.scatter(data256['Comp'].to_numpy(), data256['energy(kWh)'].to_numpy(), color='r')
plt.scatter(data512['Comp'].to_numpy(), data512['energy(kWh)'].to_numpy(), color='g')
plt.xlabel('Compression')
plt.ylabel('Energy')
plt.legend(['feat=16x16', 'feat=8x8', 'feat=4x4', 'feat=2x2'])


plt.figure()
plt.scatter(data64['MAC'].to_numpy()/data64['MAC_original'].to_numpy(), data64['energy(kWh)'].to_numpy(), color='k')
plt.scatter(data128['MAC'].to_numpy()/data128['MAC_original'].to_numpy(), data128['energy(kWh)'].to_numpy(), color='b')
plt.scatter(data256['MAC'].to_numpy()/data256['MAC_original'].to_numpy(), data256['energy(kWh)'].to_numpy(), color='r')
plt.scatter(data512['MAC'].to_numpy()/data512['MAC_original'].to_numpy(), data512['energy(kWh)'].to_numpy(), color='g')
plt.xlabel('MAC/MAC_original')
plt.ylabel('Energy')
plt.legend(['feat=16x16', 'feat=8x8', 'feat=4x4', 'feat=2x2'])

#%%
maskcp=data['Dec'].isin(['cp'])
datacp=data[maskcp]

plt.figure()
plt.scatter(datacp['Comp'].to_numpy(), datacp['MAC'].to_numpy()/datacp['MAC_original'].to_numpy(), color='k')

datastr1=datacp[datacp['Stride'].isin([1])]
datastr2=datacp[datacp['Stride'].isin([2])]

plt.figure()
plt.scatter(datastr1['Comp'].to_numpy(), datastr1['MAC'].to_numpy()/datastr1['MAC_original'].to_numpy(), color='b')
plt.scatter(datastr2['Comp'].to_numpy(), datastr2['MAC'].to_numpy()/datastr2['MAC_original'].to_numpy(), color='r')

