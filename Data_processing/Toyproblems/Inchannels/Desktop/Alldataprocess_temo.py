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
import matplotlib as mtl
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


plt.close('all')

ggplot_styles = {
    'axes.edgecolor': 'white',
    'axes.facecolor': 'EBEBEB',
    'axes.grid': True,
    'axes.grid.which': 'both',
    'axes.spines.left': False,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.spines.bottom': False,
    'grid.color': 'white',
    'grid.linewidth': '1.2',
    'xtick.color': '555555',
    'xtick.major.bottom': True,
    'xtick.minor.bottom': False,
    'ytick.color': '555555',
    'ytick.major.left': True,
    'ytick.minor.left': False,
}

plt.rcParams.update(ggplot_styles)

def custom_sort_key(s):
    # Parse the components of the string
    parts = s.split('-')
    compression_type = parts[2]
    r_value = float(parts[3][1:])
    inch = int(parts[1][4:])
    
    # Assign a numeric value to compression types to ensure correct order
    compression_order = {'factcp': 0, 'facttucker': 1, 'facttt': 2}
    
    # Return a tuple that will be used for sorting
    return (compression_order[compression_type],r_value, inch)

#%%Combine all datasets
allinfo_inch=pd.read_pickle('allinfo_in.pkl')

data_tot=pd.DataFrame.from_dict(pd.read_pickle('data_tt_dec_inch.pkl')).transpose()

sort_keys=sorted(data_tot.index, key=custom_sort_key)
data_tot = data_tot.loc[sort_keys]

data_bas=pd.DataFrame.from_dict(pd.read_pickle('data_bas_inch.pkl')).transpose()
data_bas['In_ch'] = data_bas.index.to_series().str.extract(r'inch(\d+)-wh')[0].astype(float)

# Create the new index based on the specified format
new_index = [
    f"outch{row['Out_ch']}-inch{row['In_ch']}-fact{row['Dec']}-r{row['Comp']}-wh{row['In_feat'][2]}"
    for idx, row in allinfo_inch.iterrows()
]

# Assign the new index to the DataFrame
allinfo_inch.index = new_index

#load memory measurements
mem_inch=pd.read_pickle('mem_inch (1).pkl')
mem_inch = mem_inch.loc[sort_keys]
mem_inch=mem_inch.rename(columns={'Mem':'Mem_meas'})
mem_inch.index=new_index

mem_bas=pd.read_pickle('mem_bas_inch.pkl')
mem_bas.index=data_bas.index
data_bas=pd.concat([data_bas,mem_bas], axis=1)


total_df_inch=pd.concat([data_tot,allinfo_inch, mem_inch], axis=1)

#%%dmaskdnal

data_tt=total_df_inch[total_df_inch['Dec']=='tucker']

data_inch192=data_tt[data_tt['In_ch']==192]
data_bas192=data_bas[data_bas['In_ch']==192]
data_inch256=data_tt[data_tt['In_ch']==256]
data_bas256=data_bas[data_bas['In_ch']==256]
data_inch320=data_tt[data_tt['In_ch']==320]
data_bas320=data_bas[data_bas['In_ch']==320]
data_inch384=data_tt[data_tt['In_ch']==384]
data_bas384=data_bas[data_bas['In_ch']==384]

data_inch192.loc[:, 'energy(kWh)'] = pd.to_numeric(data_inch192['energy(kWh)'], errors='coerce')
data_inch192['delta_energy(kWh)']=np.ones(len(data_inch192['energy(kWh)']))*data_bas192['energy(kWh)'].to_numpy()-data_inch192['energy(kWh)']
data_inch192.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_inch192['delta_energy(kWh)'], errors='coerce')
data_inch192['per_energy(kWh)']=data_inch192['delta_energy(kWh)']/(np.ones(len(data_inch192['energy(kWh)']))*data_bas192['energy(kWh)'].to_numpy())*100
data_inch192.loc[:,'per_energy(kWh)']=pd.to_numeric(data_inch192.loc[:,'per_energy(kWh)'],errors='coerce')
data_inch192.loc[:,'per_energy(kWh)']=np.around(data_inch192.loc[:,'per_energy(kWh)'], decimals=2)

data_inch256.loc[:, 'energy(kWh)'] = pd.to_numeric(data_inch256['energy(kWh)'], errors='coerce')
data_inch256['delta_energy(kWh)']=np.ones(len(data_inch256['energy(kWh)']))*data_bas256['energy(kWh)'].to_numpy()-data_inch256['energy(kWh)']
data_inch256.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_inch256['delta_energy(kWh)'], errors='coerce')
data_inch256['per_energy(kWh)']=data_inch256['delta_energy(kWh)']/(np.ones(len(data_inch256['energy(kWh)']))*data_bas256['energy(kWh)'].to_numpy())*100
data_inch256.loc[:,'per_energy(kWh)']=pd.to_numeric(data_inch256.loc[:,'per_energy(kWh)'],errors='coerce')
data_inch256.loc[:,'per_energy(kWh)']=np.around(data_inch256.loc[:,'per_energy(kWh)'], decimals=2)

data_inch320.loc[:, 'energy(kWh)'] = pd.to_numeric(data_inch320['energy(kWh)'], errors='coerce')
data_inch320['delta_energy(kWh)']=np.ones(len(data_inch320['energy(kWh)']))*data_bas320['energy(kWh)'].to_numpy()-data_inch320['energy(kWh)']
data_inch320.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_inch320['delta_energy(kWh)'], errors='coerce')
data_inch320['per_energy(kWh)']=data_inch320['delta_energy(kWh)']/(np.ones(len(data_inch320['energy(kWh)']))*data_bas320['energy(kWh)'].to_numpy())*100
data_inch320.loc[:,'per_energy(kWh)']=pd.to_numeric(data_inch320.loc[:,'per_energy(kWh)'],errors='coerce')
data_inch320.loc[:,'per_energy(kWh)']=np.around(data_inch320.loc[:,'per_energy(kWh)'], decimals=2)

data_inch384.loc[:, 'energy(kWh)'] = pd.to_numeric(data_inch384['energy(kWh)'], errors='coerce')
data_inch384['delta_energy(kWh)']=np.ones(len(data_inch384['energy(kWh)']))*data_bas384['energy(kWh)'].to_numpy()-data_inch384['energy(kWh)']
data_inch384.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_inch384['delta_energy(kWh)'], errors='coerce')
data_inch384['per_energy(kWh)']=data_inch384['delta_energy(kWh)']/(np.ones(len(data_inch384['energy(kWh)']))*data_bas384['energy(kWh)'].to_numpy())*100
data_inch384.loc[:,'per_energy(kWh)']=pd.to_numeric(data_inch384.loc[:,'per_energy(kWh)'],errors='coerce')
data_inch384.loc[:,'per_energy(kWh)']=np.around(data_inch384.loc[:,'per_energy(kWh)'], decimals=2)


data_tt=total_df_inch[total_df_inch['Dec']=='cp']

data_inch192=data_tt[data_tt['In_ch']==192]
data_bas192=data_bas[data_bas['In_ch']==192]
data_inch256=data_tt[data_tt['In_ch']==256]
data_bas256=data_bas[data_bas['In_ch']==256]
data_inch320=data_tt[data_tt['In_ch']==320]
data_bas320=data_bas[data_bas['In_ch']==320]
data_inch384=data_tt[data_tt['In_ch']==384]
data_bas384=data_bas[data_bas['In_ch']==384]

data_inch192.loc[:, 'energy(kWh)'] = pd.to_numeric(data_inch192['energy(kWh)'], errors='coerce')
data_inch192['delta_energy(kWh)']=np.ones(len(data_inch192['energy(kWh)']))*data_bas192['energy(kWh)'].to_numpy()-data_inch192['energy(kWh)']
data_inch192.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_inch192['delta_energy(kWh)'], errors='coerce')
data_inch192['per_energy(kWh)']=data_inch192['delta_energy(kWh)']/(np.ones(len(data_inch192['energy(kWh)']))*data_bas192['energy(kWh)'].to_numpy())*100
data_inch192.loc[:,'per_energy(kWh)']=pd.to_numeric(data_inch192.loc[:,'per_energy(kWh)'],errors='coerce')
data_inch192.loc[:,'per_energy(kWh)']=np.around(data_inch192.loc[:,'per_energy(kWh)'], decimals=2)

data_inch256.loc[:, 'energy(kWh)'] = pd.to_numeric(data_inch256['energy(kWh)'], errors='coerce')
data_inch256['delta_energy(kWh)']=np.ones(len(data_inch256['energy(kWh)']))*data_bas256['energy(kWh)'].to_numpy()-data_inch256['energy(kWh)']
data_inch256.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_inch256['delta_energy(kWh)'], errors='coerce')
data_inch256['per_energy(kWh)']=data_inch256['delta_energy(kWh)']/(np.ones(len(data_inch256['energy(kWh)']))*data_bas256['energy(kWh)'].to_numpy())*100
data_inch256.loc[:,'per_energy(kWh)']=pd.to_numeric(data_inch256.loc[:,'per_energy(kWh)'],errors='coerce')
data_inch256.loc[:,'per_energy(kWh)']=np.around(data_inch256.loc[:,'per_energy(kWh)'], decimals=2)

data_inch320.loc[:, 'energy(kWh)'] = pd.to_numeric(data_inch320['energy(kWh)'], errors='coerce')
data_inch320['delta_energy(kWh)']=np.ones(len(data_inch320['energy(kWh)']))*data_bas320['energy(kWh)'].to_numpy()-data_inch320['energy(kWh)']
data_inch320.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_inch320['delta_energy(kWh)'], errors='coerce')
data_inch320['per_energy(kWh)']=data_inch320['delta_energy(kWh)']/(np.ones(len(data_inch320['energy(kWh)']))*data_bas320['energy(kWh)'].to_numpy())*100
data_inch320.loc[:,'per_energy(kWh)']=pd.to_numeric(data_inch320.loc[:,'per_energy(kWh)'],errors='coerce')
data_inch320.loc[:,'per_energy(kWh)']=np.around(data_inch320.loc[:,'per_energy(kWh)'], decimals=2)

data_inch384.loc[:, 'energy(kWh)'] = pd.to_numeric(data_inch384['energy(kWh)'], errors='coerce')
data_inch384['delta_energy(kWh)']=np.ones(len(data_inch384['energy(kWh)']))*data_bas384['energy(kWh)'].to_numpy()-data_inch384['energy(kWh)']
data_inch384.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_inch384['delta_energy(kWh)'], errors='coerce')
data_inch384['per_energy(kWh)']=data_inch384['delta_energy(kWh)']/(np.ones(len(data_inch384['energy(kWh)']))*data_bas384['energy(kWh)'].to_numpy())*100
data_inch384.loc[:,'per_energy(kWh)']=pd.to_numeric(data_inch384.loc[:,'per_energy(kWh)'],errors='coerce')
data_inch384.loc[:,'per_energy(kWh)']=np.around(data_inch384.loc[:,'per_energy(kWh)'], decimals=2)


data_tt=total_df_inch[total_df_inch['Dec']=='tt']

data_inch192=data_tt[data_tt['In_ch']==192]
data_bas192=data_bas[data_bas['In_ch']==192]
data_inch256=data_tt[data_tt['In_ch']==256]
data_bas256=data_bas[data_bas['In_ch']==256]
data_inch320=data_tt[data_tt['In_ch']==320]
data_bas320=data_bas[data_bas['In_ch']==320]
data_inch384=data_tt[data_tt['In_ch']==384]
data_bas384=data_bas[data_bas['In_ch']==384]

data_inch192.loc[:, 'energy(kWh)'] = pd.to_numeric(data_inch192['energy(kWh)'], errors='coerce')
data_inch192['delta_energy(kWh)']=np.ones(len(data_inch192['energy(kWh)']))*data_bas192['energy(kWh)'].to_numpy()-data_inch192['energy(kWh)']
data_inch192.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_inch192['delta_energy(kWh)'], errors='coerce')
data_inch192['per_energy(kWh)']=data_inch192['delta_energy(kWh)']/(np.ones(len(data_inch192['energy(kWh)']))*data_bas192['energy(kWh)'].to_numpy())*100
data_inch192.loc[:,'per_energy(kWh)']=pd.to_numeric(data_inch192.loc[:,'per_energy(kWh)'],errors='coerce')
data_inch192.loc[:,'per_energy(kWh)']=np.around(data_inch192.loc[:,'per_energy(kWh)'], decimals=2)

data_inch256.loc[:, 'energy(kWh)'] = pd.to_numeric(data_inch256['energy(kWh)'], errors='coerce')
data_inch256['delta_energy(kWh)']=np.ones(len(data_inch256['energy(kWh)']))*data_bas256['energy(kWh)'].to_numpy()-data_inch256['energy(kWh)']
data_inch256.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_inch256['delta_energy(kWh)'], errors='coerce')
data_inch256['per_energy(kWh)']=data_inch256['delta_energy(kWh)']/(np.ones(len(data_inch256['energy(kWh)']))*data_bas256['energy(kWh)'].to_numpy())*100
data_inch256.loc[:,'per_energy(kWh)']=pd.to_numeric(data_inch256.loc[:,'per_energy(kWh)'],errors='coerce')
data_inch256.loc[:,'per_energy(kWh)']=np.around(data_inch256.loc[:,'per_energy(kWh)'], decimals=2)

data_inch320.loc[:, 'energy(kWh)'] = pd.to_numeric(data_inch320['energy(kWh)'], errors='coerce')
data_inch320['delta_energy(kWh)']=np.ones(len(data_inch320['energy(kWh)']))*data_bas320['energy(kWh)'].to_numpy()-data_inch320['energy(kWh)']
data_inch320.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_inch320['delta_energy(kWh)'], errors='coerce')
data_inch320['per_energy(kWh)']=data_inch320['delta_energy(kWh)']/(np.ones(len(data_inch320['energy(kWh)']))*data_bas320['energy(kWh)'].to_numpy())*100
data_inch320.loc[:,'per_energy(kWh)']=pd.to_numeric(data_inch320.loc[:,'per_energy(kWh)'],errors='coerce')
data_inch320.loc[:,'per_energy(kWh)']=np.around(data_inch320.loc[:,'per_energy(kWh)'], decimals=2)

data_inch384.loc[:, 'energy(kWh)'] = pd.to_numeric(data_inch384['energy(kWh)'], errors='coerce')
data_inch384['delta_energy(kWh)']=np.ones(len(data_inch384['energy(kWh)']))*data_bas384['energy(kWh)'].to_numpy()-data_inch384['energy(kWh)']
data_inch384.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_inch384['delta_energy(kWh)'], errors='coerce')
data_inch384['per_energy(kWh)']=data_inch384['delta_energy(kWh)']/(np.ones(len(data_inch384['energy(kWh)']))*data_bas384['energy(kWh)'].to_numpy())*100
data_inch384.loc[:,'per_energy(kWh)']=pd.to_numeric(data_inch384.loc[:,'per_energy(kWh)'],errors='coerce')
data_inch384.loc[:,'per_energy(kWh)']=np.around(data_inch384.loc[:,'per_energy(kWh)'], decimals=2)


#%%


#%%Let's create plots to show difference in in_channels

data_tt=total_df_inch[total_df_inch['Dec']=='tucker']

data_inch192=data_tt[data_tt['In_ch']==192]
data_bas192=data_bas[data_bas['In_ch']==192]
data_inch256=data_tt[data_tt['In_ch']==256]
data_bas256=data_bas[data_bas['In_ch']==256]
data_inch320=data_tt[data_tt['In_ch']==320]
data_bas320=data_bas[data_bas['In_ch']==320]
data_inch384=data_tt[data_tt['In_ch']==384]
data_bas384=data_bas[data_bas['In_ch']==384]

data_inch192.loc[:, 'energy(kWh)'] = pd.to_numeric(data_inch192['energy(kWh)'], errors='coerce')
data_inch192['delta_energy(kWh)']=np.ones(len(data_inch192['energy(kWh)']))*data_bas192['energy(kWh)'].to_numpy()-data_inch192['energy(kWh)']
data_inch192.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_inch192['delta_energy(kWh)'], errors='coerce')
data_inch192['per_energy(kWh)']=data_inch192['delta_energy(kWh)']/(np.ones(len(data_inch192['energy(kWh)']))*data_bas192['energy(kWh)'].to_numpy())*100
data_inch192.loc[:,'per_energy(kWh)']=pd.to_numeric(data_inch192.loc[:,'per_energy(kWh)'],errors='coerce')
data_inch192.loc[:,'per_energy(kWh)']=np.around(data_inch192.loc[:,'per_energy(kWh)'], decimals=2)
z192 = np.polyfit(data_inch192['Comp'], data_inch192['delta_energy(kWh)'], 2)
p192 = np.poly1d(z192)
xp192 = np.linspace(0.1, 0.9, 100)


data_inch256.loc[:, 'energy(kWh)'] = pd.to_numeric(data_inch256['energy(kWh)'], errors='coerce')
data_inch256['delta_energy(kWh)']=np.ones(len(data_inch256['energy(kWh)']))*data_bas256['energy(kWh)'].to_numpy()-data_inch256['energy(kWh)']
data_inch256.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_inch256['delta_energy(kWh)'], errors='coerce')
data_inch256['per_energy(kWh)']=data_inch256['delta_energy(kWh)']/(np.ones(len(data_inch256['energy(kWh)']))*data_bas256['energy(kWh)'].to_numpy())*100
data_inch256.loc[:,'per_energy(kWh)']=pd.to_numeric(data_inch256.loc[:,'per_energy(kWh)'],errors='coerce')
data_inch256.loc[:,'per_energy(kWh)']=np.around(data_inch256.loc[:,'per_energy(kWh)'], decimals=2)

z256 = np.polyfit(data_inch256['Comp'], data_inch256['delta_energy(kWh)'], 2)
p256 = np.poly1d(z256)
xp256 = np.linspace(0.1, 0.9, 100)

data_inch320.loc[:, 'energy(kWh)'] = pd.to_numeric(data_inch320['energy(kWh)'], errors='coerce')
data_inch320['delta_energy(kWh)']=np.ones(len(data_inch320['energy(kWh)']))*data_bas320['energy(kWh)'].to_numpy()-data_inch320['energy(kWh)']
data_inch320.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_inch320['delta_energy(kWh)'], errors='coerce')
data_inch320['per_energy(kWh)']=data_inch320['delta_energy(kWh)']/(np.ones(len(data_inch320['energy(kWh)']))*data_bas320['energy(kWh)'].to_numpy())*100
data_inch320.loc[:,'per_energy(kWh)']=pd.to_numeric(data_inch320.loc[:,'per_energy(kWh)'],errors='coerce')
data_inch320.loc[:,'per_energy(kWh)']=np.around(data_inch320.loc[:,'per_energy(kWh)'], decimals=2)

z320 = np.polyfit(data_inch320['Comp'], data_inch320['delta_energy(kWh)'], 2)
p320 = np.poly1d(z320)
xp320 = np.linspace(0.1, 0.9, 100)

data_inch384.loc[:, 'energy(kWh)'] = pd.to_numeric(data_inch384['energy(kWh)'], errors='coerce')
data_inch384['delta_energy(kWh)']=np.ones(len(data_inch384['energy(kWh)']))*data_bas384['energy(kWh)'].to_numpy()-data_inch384['energy(kWh)']
data_inch384.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_inch384['delta_energy(kWh)'], errors='coerce')
data_inch384['per_energy(kWh)']=data_inch384['delta_energy(kWh)']/(np.ones(len(data_inch384['energy(kWh)']))*data_bas384['energy(kWh)'].to_numpy())*100
data_inch384.loc[:,'per_energy(kWh)']=pd.to_numeric(data_inch384.loc[:,'per_energy(kWh)'],errors='coerce')
data_inch384.loc[:,'per_energy(kWh)']=np.around(data_inch384.loc[:,'per_energy(kWh)'], decimals=2)

z384 = np.polyfit(data_inch384['Comp'], data_inch384['delta_energy(kWh)'], 2)
p384 = np.poly1d(z384)
xp384 = np.linspace(0.1, 0.9, 100)

plt.figure()
plt.scatter(data_inch192['Comp'], data_inch192['delta_energy(kWh)'],label='In_ch=192', color='c')
plt.plot(xp192, p192(xp192), 'c' + '-')
plt.scatter(data_inch256['Comp'], data_inch256['delta_energy(kWh)'],label='In_ch=256', color='m')
plt.plot(xp256, p256(xp256), 'm' + '-')
plt.scatter(data_inch320['Comp'], data_inch320['delta_energy(kWh)'],label='In_ch=320', color='k')
plt.plot(xp320, p320(xp320), 'k' + '-')
plt.scatter(data_inch384['Comp'], data_inch384['delta_energy(kWh)'],label='In_ch=384',color='y')
plt.plot(xp384, p384(xp384), 'y' + '-')
plt.title('Energy saved per number of input channels for Tuck')
plt.xlabel('Compression')
plt.ylabel('Energy saved [%]')
plt.legend()

#%%
data_inch192=data_tt[data_tt['In_ch']==192]
data_bas192=data_bas[data_bas['In_ch']==192]
data_inch256=data_tt[data_tt['In_ch']==256]
data_bas256=data_bas[data_bas['In_ch']==256]
data_inch320=data_tt[data_tt['In_ch']==320]
data_bas320=data_bas[data_bas['In_ch']==320]
data_inch384=data_tt[data_tt['In_ch']==384]
data_bas384=data_bas[data_bas['In_ch']==384]

data_inch192.loc[:, 'energy(kWh)'] = pd.to_numeric(data_inch192['energy(kWh)'], errors='coerce')
data_inch192['delta_energy(kWh)']=np.ones(len(data_inch192['energy(kWh)']))*data_bas192['energy(kWh)'].to_numpy()-data_inch192['energy(kWh)']
data_inch192.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_inch192['delta_energy(kWh)'], errors='coerce')
data_inch192['per_energy(kWh)']=data_inch192['delta_energy(kWh)']/(np.ones(len(data_inch192['energy(kWh)']))*data_bas192['energy(kWh)'].to_numpy())*100
data_inch192.loc[:,'per_energy(kWh)']=pd.to_numeric(data_inch192.loc[:,'per_energy(kWh)'],errors='coerce')
data_inch192.loc[:,'per_energy(kWh)']=np.around(data_inch192.loc[:,'per_energy(kWh)'], decimals=2)
z192 = np.polyfit(data_inch192['Comp'], (data_inch192['MAC_original']-data_inch192['MAC'])/data_inch192['MAC_original']*100, 2)
p192 = np.poly1d(z192)
xp192 = np.linspace(0.1, 0.9, 100)


data_inch256.loc[:, 'energy(kWh)'] = pd.to_numeric(data_inch256['energy(kWh)'], errors='coerce')
data_inch256['delta_energy(kWh)']=np.ones(len(data_inch256['energy(kWh)']))*data_bas256['energy(kWh)'].to_numpy()-data_inch256['energy(kWh)']
data_inch256.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_inch256['delta_energy(kWh)'], errors='coerce')
data_inch256['per_energy(kWh)']=data_inch256['delta_energy(kWh)']/(np.ones(len(data_inch256['energy(kWh)']))*data_bas256['energy(kWh)'].to_numpy())*100
data_inch256.loc[:,'per_energy(kWh)']=pd.to_numeric(data_inch256.loc[:,'per_energy(kWh)'],errors='coerce')
data_inch256.loc[:,'per_energy(kWh)']=np.around(data_inch256.loc[:,'per_energy(kWh)'], decimals=2)

z256 = np.polyfit(data_inch256['Comp'], (data_inch256['MAC_original']-data_inch256['MAC'])/data_inch256['MAC_original']*100, 2)
p256 = np.poly1d(z256)
xp256 = np.linspace(0.1, 0.9, 100)

data_inch320.loc[:, 'energy(kWh)'] = pd.to_numeric(data_inch320['energy(kWh)'], errors='coerce')
data_inch320['delta_energy(kWh)']=np.ones(len(data_inch320['energy(kWh)']))*data_bas320['energy(kWh)'].to_numpy()-data_inch320['energy(kWh)']
data_inch320.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_inch320['delta_energy(kWh)'], errors='coerce')
data_inch320['per_energy(kWh)']=data_inch320['delta_energy(kWh)']/(np.ones(len(data_inch320['energy(kWh)']))*data_bas320['energy(kWh)'].to_numpy())*100
data_inch320.loc[:,'per_energy(kWh)']=pd.to_numeric(data_inch320.loc[:,'per_energy(kWh)'],errors='coerce')
data_inch320.loc[:,'per_energy(kWh)']=np.around(data_inch320.loc[:,'per_energy(kWh)'], decimals=2)

z320 = np.polyfit(data_inch320['Comp'], (data_inch320['MAC_original']-data_inch320['MAC'])/data_inch320['MAC_original']*100, 2)
p320 = np.poly1d(z320)
xp320 = np.linspace(0.1, 0.9, 100)

data_inch384.loc[:, 'energy(kWh)'] = pd.to_numeric(data_inch384['energy(kWh)'], errors='coerce')
data_inch384['delta_energy(kWh)']=np.ones(len(data_inch384['energy(kWh)']))*data_bas384['energy(kWh)'].to_numpy()-data_inch384['energy(kWh)']
data_inch384.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_inch384['delta_energy(kWh)'], errors='coerce')
data_inch384['per_energy(kWh)']=data_inch384['delta_energy(kWh)']/(np.ones(len(data_inch384['energy(kWh)']))*data_bas384['energy(kWh)'].to_numpy())*100
data_inch384.loc[:,'per_energy(kWh)']=pd.to_numeric(data_inch384.loc[:,'per_energy(kWh)'],errors='coerce')
data_inch384.loc[:,'per_energy(kWh)']=np.around(data_inch384.loc[:,'per_energy(kWh)'], decimals=2)

z384 = np.polyfit(data_inch384['Comp'], (data_inch384['MAC_original']-data_inch384['MAC'])/data_inch384['MAC_original']*100, 2)
p384 = np.poly1d(z384)
xp384 = np.linspace(0.1, 0.9, 100)

plt.figure()
plt.scatter(data_inch192['Comp'], (data_inch192['MAC_original']-data_inch192['MAC'])/data_inch192['MAC_original']*100,label='In_ch=192', color='c')
plt.plot(xp192, p192(xp192), 'c' + '-')
plt.scatter(data_inch256['Comp'], (data_inch256['MAC_original']-data_inch256['MAC'])/data_inch256['MAC_original']*100,label='In_ch=256', color='m')
plt.plot(xp256, p256(xp256), 'm' + '-')
plt.scatter(data_inch320['Comp'], (data_inch320['MAC_original']-data_inch320['MAC'])/data_inch320['MAC_original']*100,label='In_ch=320', color='k')
plt.plot(xp320, p320(xp320), 'k' + '-')
plt.scatter(data_inch384['Comp'], (data_inch384['MAC_original']-data_inch384['MAC'])/data_inch384['MAC_original']*100,label='In_ch=384',color='y')
plt.plot(xp384, p384(xp384), 'y' + '-')
plt.title('MAC operations saved (%) for Tuck')
plt.xlabel('Compression')
plt.ylabel('MACs saved [%]')
plt.legend()

# save_path = "inchall_reg.pkl"
# with open(save_path, 'wb') as f:
#     pickle.dump(total_df_inch, f)