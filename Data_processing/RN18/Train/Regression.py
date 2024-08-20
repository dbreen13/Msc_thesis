#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:35:15 2024

@author: dbreen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%Import data

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



#%%Stride=1

#Layer 63
lay_63=data[data['Layer'].isin([63])]
lay_63=lay_63[lay_63['Dec'].isin(['cp'])]

energy=lay_63['energy(kWh)'].to_numpy(dtype=float)
compr=lay_63['Comp'].to_numpy(dtype=float)

z = np.polyfit(compr, energy ,4)
p = np.poly1d(z)
t=np.linspace(0.1,0.9,100)

plt.figure(),
plt.scatter(compr, energy, color='C0')
plt.plot(t,np.ones(100)*baseline['energy(kWh)'].to_numpy()[0], label='baseline', color='C2')
plt.plot(t,p(t), label='feat=2x2', color='C0')


#Layer 47
lay_63=data[data['Layer'].isin([47])]
lay_63=lay_63[lay_63['Dec'].isin(['cp'])]

energy=lay_63['energy(kWh)'].to_numpy(dtype=float)
compr=lay_63['Comp'].to_numpy(dtype=float)

z = np.polyfit(compr, energy ,4)
p = np.poly1d(z)
t=np.linspace(0.1,0.9,100)

plt.scatter(compr, energy, color='C1')
plt.plot(t,p(t), label='feat=4x4', color='C1')
plt.legend()

#Layer 28
lay_63=data[data['Layer'].isin([28])]
lay_63=lay_63[lay_63['Dec'].isin(['cp'])]

energy=lay_63['energy(kWh)'].to_numpy(dtype=float)
compr=lay_63['Comp'].to_numpy(dtype=float)

z = np.polyfit(compr, energy ,4)
p = np.poly1d(z)
t=np.linspace(0.1,0.9,100)

plt.scatter(compr, energy, color='C4')
plt.plot(t,p(t), label='feat=8x8', color='C4')
plt.legend()

#Layer 6
lay_63=data[data['Layer'].isin([6])]
lay_63=lay_63[lay_63['Dec'].isin(['cp'])]

energy=lay_63['energy(kWh)'].to_numpy(dtype=float)
compr=lay_63['Comp'].to_numpy(dtype=float)

z = np.polyfit(compr, energy ,4)
p = np.poly1d(z)
t=np.linspace(0.1,0.9,100)

plt.scatter(compr, energy, color='C3')
plt.plot(t,p(t), label='feat=16x16', color='C3')
plt.xlabel('Compression')
plt.ylabel('Energy consumed (kWh)')
plt.title('Energy consumption feature sizes, for stride=1')

plt.legend()

#%%Stride=2

#Layer 57
lay_63=data[data['Layer'].isin([57])]
lay_63=lay_63[lay_63['Dec'].isin(['cp'])]

energy=lay_63['energy(kWh)'].to_numpy(dtype=float)
compr=lay_63['Comp'].to_numpy(dtype=float)

z = np.polyfit(compr, energy ,4)
p = np.poly1d(z)
t=np.linspace(0.1,0.9,100)

plt.figure(),
plt.scatter(compr, energy, color='C0')
plt.plot(t,np.ones(100)*baseline['energy(kWh)'].to_numpy()[0], label='baseline', color='C2')
plt.plot(t,p(t), label='feat=2x2', color='C0')


#Layer 7
lay_63=data[data['Layer'].isin([51])]
lay_63=lay_63[lay_63['Dec'].isin(['cp'])]

energy=lay_63['energy(kWh)'].to_numpy(dtype=float)
compr=lay_63['Comp'].to_numpy(dtype=float)

z = np.polyfit(compr, energy ,4)
p = np.poly1d(z)
t=np.linspace(0.1,0.9,100)

plt.scatter(compr, energy, color='C1')
plt.plot(t,p(t), label='feat=4x4', color='C1')
plt.legend()

#Layer 35
lay_63=data[data['Layer'].isin([35])]
lay_63=lay_63[lay_63['Dec'].isin(['cp'])]

energy=lay_63['energy(kWh)'].to_numpy(dtype=float)
compr=lay_63['Comp'].to_numpy(dtype=float)

z = np.polyfit(compr, energy ,4)
p = np.poly1d(z)
t=np.linspace(0.1,0.9,100)

plt.scatter(compr, energy, color='C4')
plt.plot(t,p(t), label='feat=8x8', color='C4')
plt.legend()

#Layer 19
lay_63=data[data['Layer'].isin([19])]
lay_63=lay_63[lay_63['Dec'].isin(['cp'])]

energy=lay_63['energy(kWh)'].to_numpy(dtype=float)
compr=lay_63['Comp'].to_numpy(dtype=float)

z = np.polyfit(compr, energy ,4)
p = np.poly1d(z)
t=np.linspace(0.1,0.9,100)

plt.scatter(compr, energy, color='C3')
plt.plot(t,p(t), label='feat=16x16', color='C3')
plt.xlabel('Compression')
plt.ylabel('Energy consumed (kWh)')
plt.title('Energy consumption feature sizes, for stride=2')
plt.legend()


#%%Try MACS/MACS_orgiginal
i=datacp[data.Layer==51].index
datacp=datacp.drop(i)

data_cp_mac=datacp['MAC'].to_numpy()

data_cp_origial_mac=datacp['MAC_original'].to_numpy()
ratio_mac=data_cp_mac/data_cp_origial_mac

data_energy=datacp['energy(kWh)'].to_numpy(dtype=float)

# z = np.polyfit(ratio_mac, data_energy ,3)
# p = np.poly1d(z)
# t=np.linspace(0.0,np.max(ratio_mac),100)

plt.figure()
plt.scatter(ratio_mac,data_energy)
#plt.plot(t,p(t), label='feat=16x16', color='C3')
plt.ylabel('Energy (kWh)')
plt.xlabel('MACs/MACs_original')
