#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:22:07 2024

@author: dbreen
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_pickle('allinfo.pkl')

data_eng=pd.read_pickle('dataset_final.pkl').transpose()
baseline=data_eng.iloc[:1]
data_eng=data_eng[1:]
index=data_eng.index.to_numpy()
data.reset_index(drop=True, inplace=True)
data_eng.reset_index(drop=True, inplace=True)

data=pd.concat([data_eng,data], axis=1)
# data.set_index(index)

data['MAC_original']=data['MAC_original']*128

maskcp=data['Dec'].isin(['cp'])
datacp=data[maskcp]

memory=pd.read_pickle('df_cp_mem.pkl')
memory=memory['Memory']
memory.reset_index
data.reset_index
data=pd.concat([data,memory], axis=1)

#%%cp decomposition
maskcp=data['Dec'].isin(['cp'])
datacp=data[maskcp]

data64=datacp[datacp['In_feat'].apply(lambda x: np.array_equal(x, np.array([128, 64, 16, 16])))]
data64=data64[data64['Out_feat'].apply(lambda x: np.array_equal(x, np.array([128, 64, 16, 16])))]


data128=datacp[datacp['In_feat'].apply(lambda x: np.array_equal(x, np.array([128, 128, 8, 8])))]
data128=data128[data128['Out_feat'].apply(lambda x: np.array_equal(x, np.array([128, 128, 8,8])))]

data256=datacp[datacp['In_feat'].apply(lambda x: np.array_equal(x, np.array([128, 256, 4, 4])))]
data256=data256[data256['Out_feat'].apply(lambda x: np.array_equal(x, np.array([128, 256, 4, 4])))]

data512=datacp[datacp['In_feat'].apply(lambda x: np.array_equal(x, np.array([128, 512, 2, 2])))]
data512=data512[data512['Out_feat'].apply(lambda x: np.array_equal(x, np.array([128, 512, 2, 2])))]


plt.figure()
plt.scatter(data64['Comp'].to_numpy(), data64['Memory'].to_numpy(dtype=int), color='k')
plt.scatter(data128['Comp'].to_numpy(), data128['Memory'].to_numpy(dtype=int), color='b')
plt.scatter(data256['Comp'].to_numpy(), data256['Memory'].to_numpy(dtype=int), color='r')
plt.scatter(data512['Comp'].to_numpy(), data512['Memory'].to_numpy(dtype=int), color='g')
plt.xlabel('Compression')
plt.ylabel('Memory[MB]')
plt.title('Memory vs compression for each feature input, stride=1')
plt.legend(['feat=16x16', 'feat=8x8', 'feat=4x4', 'feat=2x2'])

#%%
for in_ch in [64,128,256]: 
        datastr_cp=datacp[datacp['Stride'].isin([2])]
        data_in=datastr_cp[datastr_cp['In_ch'].isin([in_ch])]
        data_kern2=data_in[data_in['Kernel'].isin([3])]
        
        MAC_str2=data_kern2['Memory'].to_numpy()
        
        
        datastr_cp=datacp[datacp['Stride'].isin([1])]
        data_in=datastr_cp[datastr_cp['In_ch'].isin([in_ch])]
        data_kern1=data_in[data_in['Kernel'].isin([3])]
        
        MAC_str1=data_kern1['Memory'].to_numpy()
        
        plt.figure()
        plt.scatter(data_kern2['Comp'].to_numpy(), MAC_str2,color='b')
        plt.scatter(data_kern1['Comp'].to_numpy(), MAC_str1)
        plt.legend(['Stride=1', 'Stride=2'])
        plt.ylim([0,1000])
        plt.title(f'Influence stride on MAC for in_ch={in_ch}')
        plt.xlabel('Compression')
        plt.ylabel('Memory [Mb]')
        plt.savefig(f'Figures/Stride/stride_inch{in_ch}.png')
#%%        
#Layer 63
lay_63=data[data['Layer'].isin([63])]
lay_63=lay_63[lay_63['Dec'].isin(['cp'])]

energy=lay_63['Memory'].to_numpy(dtype=float)
compr=lay_63['Comp'].to_numpy(dtype=float)

z = np.polyfit(compr, energy ,4)
p = np.poly1d(z)
t=np.linspace(0.1,0.9,100)

plt.figure(),
plt.scatter(compr, energy, color='C0')
#plt.plot(t,np.ones(100)*baseline['energy(kWh)'].to_numpy()[0], label='baseline', color='C2')
plt.plot(t,p(t), label='Layer 54', color='C0')

lay_63=data[data['Layer'].isin([51])]
lay_63=lay_63[lay_63['Dec'].isin(['cp'])]

energy=lay_63['Memory'].to_numpy(dtype=float)
compr=lay_63['Comp'].to_numpy(dtype=float)

z = np.polyfit(compr, energy ,4)
p = np.poly1d(z)
t=np.linspace(0.1,0.9,100)

plt.scatter(compr, energy, color='C2')
#plt.plot(t,np.ones(100)*baseline['energy(kWh)'].to_numpy()[0], label='baseline', color='C2')
plt.plot(t,p(t), label='Layer 51', color='C2')
plt.legend()
plt.xlabel('Compression')
plt.ylabel('Memory (Mb)')
plt.title('Memory vs compression for both layers')

#%%        
#Layer 63
lay_63=data[data['Layer'].isin([63])]
lay_63=lay_63[lay_63['Dec'].isin(['cp'])]

energy=lay_63['MAC'].to_numpy(dtype=float)
compr=lay_63['Comp'].to_numpy(dtype=float)

z = np.polyfit(compr, energy ,4)
p = np.poly1d(z)
t=np.linspace(0.1,0.9,100)

plt.figure(),
plt.scatter(compr, energy, color='C0')
#plt.plot(t,np.ones(100)*baseline['energy(kWh)'].to_numpy()[0], label='baseline', color='C2')
plt.plot(t,p(t), label='Layer 54', color='C0')

lay_63=data[data['Layer'].isin([51])]
lay_63=lay_63[lay_63['Dec'].isin(['cp'])]

energy=lay_63['MAC'].to_numpy(dtype=float)
compr=lay_63['Comp'].to_numpy(dtype=float)

z = np.polyfit(compr, energy ,4)
p = np.poly1d(z)
t=np.linspace(0.1,0.9,100)

plt.scatter(compr, energy, color='C2')
#plt.plot(t,np.ones(100)*baseline['energy(kWh)'].to_numpy()[0], label='baseline', color='C2')
plt.plot(t,p(t), label='Layer 51', color='C2')
plt.legend()
plt.xlabel('Compression')
plt.ylabel('MACs')
plt.title('MAC vs compression for both layers')


        