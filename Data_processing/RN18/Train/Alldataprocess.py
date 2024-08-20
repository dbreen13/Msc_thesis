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
import matplotlib.lines as mlines


plt.close('all')

ggplot_styles = {
    'axes.edgecolor': 'white',
    'axes.facecolor': '#EBEBEB',
    'axes.grid': True,
    'axes.grid.which': 'both',
    'axes.spines.left': False,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.spines.bottom': False,
    'grid.color': 'white',
    'grid.linewidth': 1.2,
    'xtick.color': '#555555',
    'xtick.major.bottom': True,
    'xtick.minor.bottom': False,
    'ytick.color': '#555555',
    'ytick.major.left': True,
    'ytick.minor.left': False,
}

plt.rcParams.update(ggplot_styles)


def get_color():
    for item in ['r', 'g', 'b', 'c', 'm', 'y', 'k']:
        yield item

#%%Combine all datasets
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


data['new']=(2*data['Out_ch']-data['In_ch'])*data['Stride']*data['Kernel']
data.loc[:,'new']=pd.to_numeric(data.loc[:,'new'],errors='coerce')
data.loc[:,'new']=np.around(data.loc[:,'new'], decimals=2)


#%%Seperate data per method
maskcp=data['Dec'].isin(['cp'])
datacp=data.copy()[maskcp]
datacp=datacp[datacp['Layer'].isin([63, 57,51,47,41,35,28,25,19,6])]

masktt=data['Dec'].isin(['tt'])
datatt=data.copy()[masktt]
datatt=datatt[datatt['Layer'].isin([63, 57,51,47,41,35,28,25,19,6])]

masktucker=data['Dec'].isin(['tucker'])
datatucker=data.copy()[masktucker]
datatucker=datatucker[datatucker['Layer'].isin([63, 57,51,47,41,35,28,25,19,6])]

#%% Heatmaps for the differnt methods
# Convert the 'energy(kWh)' column to float, errors='coerce' will convert non-convertible values to NaN

sns.set(font_scale=1.1)

# Custom sequential colormap based on '#4477aa'
colors = [
    (0.0, "white"),
    (0.02, "#f0f5fc"),  # very very light blue
    (0.05, "#e1ebf7"),  # very light blue
    (0.10, "#c3d8ef"),  # light blue
    (0.15, "#a6c8e5"),  # light-medium blue
    (0.25, "#6ea4d2"),  # medium blue
    (0.50, "#4477aa"),  # blue
    (0.75, "#2e5781"),  # darker blue
    (1.0, "#1b3a55")    # darkest blue
]

n_bins = 100  # Discretizes the interpolation into bins
cmap_name = 'custom_sequential'
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


# def prepare_data(df, baseline):
#     df.loc[:, 'energy(kWh)'] = pd.to_numeric(df['energy(kWh)'], errors='coerce')
#     df['delta_energy(kWh)'] = baseline['energy(kWh)'].to_numpy() - df['energy(kWh)']
#     df.loc[:, 'delta_energy(kWh)'] = pd.to_numeric(df['delta_energy(kWh)'], errors='coerce')
#     df['per_energy(kWh)'] = df['delta_energy(kWh)'] / baseline['energy(kWh)'].to_numpy() * 100
#     df.loc[:, 'per_energy(kWh)'] = pd.to_numeric(df['per_energy(kWh)'], errors='coerce')
#     df.loc[:, 'per_energy(kWh)'] = np.around(df['per_energy(kWh)'], decimals=2)
#     return df

# datacp = prepare_data(datacp, baseline)
# datatucker = prepare_data(datatucker, baseline)
# datatt = prepare_data(datatt, baseline)

plt.close('all')
datacp.loc[:, 'energy(kWh)'] = pd.to_numeric(datacp['energy(kWh)'], errors='coerce')
datacp['delta_energy(kWh)']=np.ones(len(datacp['energy(kWh)']))*baseline['energy(kWh)'].to_numpy()-datacp['energy(kWh)']
datacp.loc[:,'delta_energy(kWh)']=pd.to_numeric(datacp['delta_energy(kWh)'], errors='coerce')

datacp['per_energy(kWh)']=datacp['delta_energy(kWh)']/(np.ones(len(datacp['energy(kWh)']))*baseline['energy(kWh)'].to_numpy())*100
datacp.loc[:,'per_energy(kWh)']=pd.to_numeric(datacp.loc[:,'per_energy(kWh)'],errors='coerce')
datacp.loc[:,'per_energy(kWh)']=np.around(datacp.loc[:,'per_energy(kWh)'], decimals=2)



heatmap_data_cp = datacp.pivot( "Comp","Layer", "Memcalc").sort_index(ascending=False)

datatucker.loc[:, 'energy(kWh)'] = pd.to_numeric(datatucker['energy(kWh)'], errors='coerce')
datatucker['delta_energy(kWh)']=np.ones(len(datatucker['energy(kWh)']))*baseline['energy(kWh)'].to_numpy()-datatucker['energy(kWh)']
datatucker.loc[:,'delta_energy(kWh)']=pd.to_numeric(datatucker['delta_energy(kWh)'], errors='coerce')
datatucker['per_energy(kWh)']=datatucker['delta_energy(kWh)']/(np.ones(len(datatucker['energy(kWh)']))*baseline['energy(kWh)'].to_numpy())*100
datatucker.loc[:,'per_energy(kWh)']=pd.to_numeric(datatucker.loc[:,'per_energy(kWh)'],errors='coerce')
datatucker.loc[:,'per_energy(kWh)']=np.around(datatucker.loc[:,'per_energy(kWh)'], decimals=2)


heatmap_data_tuck = datatucker.pivot( "Comp","Layer", "Memcalc").sort_index(ascending=False)


datatt.loc[:, 'energy(kWh)'] = pd.to_numeric(datatt['energy(kWh)'], errors='coerce')
datatt['delta_energy(kWh)']=np.ones(len(datatt['energy(kWh)']))*baseline['energy(kWh)'].to_numpy()-datatt['energy(kWh)']
datatt.loc[:,'delta_energy(kWh)']=pd.to_numeric(datatt['delta_energy(kWh)'], errors='coerce')
datatt['per_energy(kWh)']=datatt['delta_energy(kWh)']/(np.ones(len(datatt['energy(kWh)']))*baseline['energy(kWh)'].to_numpy())*100
datatt.loc[:,'per_energy(kWh)']=pd.to_numeric(datatt.loc[:,'per_energy(kWh)'],errors='coerce')
datatt.loc[:,'per_energy(kWh)']=np.around(datatt.loc[:,'per_energy(kWh)'], decimals=2)


heatmap_data_tt = datatt.pivot( "Comp","Layer", "Memcalc").sort_index(ascending=False)

# Concatenate all data
all_data = pd.concat([datacp, datatt, datatucker])

# Calculate vmin and vmax
vmin = all_data['Memcalc'].min()
vmax = all_data['Memcalc'].max()

# Create the heatmap for Tucker Decomposition
plt.figure(figsize=(5, 5))
sns.heatmap(heatmap_data_tuck, annot=False, fmt=".2e", cmap=cmap, cbar_kws={'label': 'Added memory (# params)'}, vmin=vmin, vmax=vmax, annot_kws={"size": 12})
plt.title('Additional memory Tucker', pad=20)
plt.xlabel('Layer')  
plt.ylabel('Compression Ratio')
plt.show()

# Create the heatmap for CP Decomposition
plt.figure(figsize=(5, 5))
sns.heatmap(heatmap_data_cp, annot=False, fmt=".2e", cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={'label': 'Added memory (# params)'}, annot_kws={"size": 12})
plt.title('Additional memory CP', pad=20)
plt.xlabel('Layer')  
plt.ylabel('Compression Ratio')
plt.show()

# Create the heatmap for TT Decomposition
plt.figure(figsize=(5, 5))
sns.heatmap(heatmap_data_tt, annot=False, fmt=".2e", cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={'label': 'Added memory (# params)'}, annot_kws={"size": 12})
plt.title('Additional memory TT', pad=20)
plt.xlabel('Layer')  
plt.ylabel('Compression Ratio')
plt.show()


#%%
sns.set(font_scale=1.1)

# Custom sequential colormap based on '#4477aa'
colors = [
    (0.0, "white"),
    (0.02, "#f0f7fc"),  # very very light blue
    (0.05, "#dfeef7"),  # very light blue
    (0.10, "#cce4f0"),  # light blue
    (0.15, "#b3d8e8"),  # light-medium blue
    (0.25, "#99ccea"),  # medium blue
    (0.50, "#66a3d2"),  # green
    (0.75, "#4477aa"),  # base blue
    (1.0, "#1b3a55")    # darkest blue
]

n_bins = 100  # Discretizes the interpolation into bins
cmap_name = 'custom_sequential'
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


# def prepare_data(df, baseline):
#     df.loc[:, 'energy(kWh)'] = pd.to_numeric(df['energy(kWh)'], errors='coerce')
#     df['delta_energy(kWh)'] = baseline['energy(kWh)'].to_numpy() - df['energy(kWh)']
#     df.loc[:, 'delta_energy(kWh)'] = pd.to_numeric(df['delta_energy(kWh)'], errors='coerce')
#     df['per_energy(kWh)'] = df['delta_energy(kWh)'] / baseline['energy(kWh)'].to_numpy() * 100
#     df.loc[:, 'per_energy(kWh)'] = pd.to_numeric(df['per_energy(kWh)'], errors='coerce')
#     df.loc[:, 'per_energy(kWh)'] = np.around(df['per_energy(kWh)'], decimals=2)
#     return df

# datacp = prepare_data(datacp, baseline)
# datatucker = prepare_data(datatucker, baseline)
# datatt = prepare_data(datatt, baseline)

plt.close('all')
datacp.loc[:, 'energy(kWh)'] = pd.to_numeric(datacp['energy(kWh)'], errors='coerce')
datacp['delta_energy(kWh)']=np.ones(len(datacp['energy(kWh)']))*baseline['energy(kWh)'].to_numpy()-datacp['energy(kWh)']
datacp.loc[:,'delta_energy(kWh)']=pd.to_numeric(datacp['delta_energy(kWh)'], errors='coerce')

datacp['per_energy(kWh)']=datacp['delta_energy(kWh)']/(np.ones(len(datacp['energy(kWh)']))*baseline['energy(kWh)'].to_numpy())*100
datacp.loc[:,'per_energy(kWh)']=pd.to_numeric(datacp.loc[:,'per_energy(kWh)'],errors='coerce')
datacp.loc[:,'per_energy(kWh)']=np.around(datacp.loc[:,'per_energy(kWh)'], decimals=2)




datatucker.loc[:, 'energy(kWh)'] = pd.to_numeric(datatucker['energy(kWh)'], errors='coerce')
datatucker['delta_energy(kWh)']=np.ones(len(datatucker['energy(kWh)']))*baseline['energy(kWh)'].to_numpy()-datatucker['energy(kWh)']
datatucker.loc[:,'delta_energy(kWh)']=pd.to_numeric(datatucker['delta_energy(kWh)'], errors='coerce')
datatucker['per_energy(kWh)']=datatucker['delta_energy(kWh)']/(np.ones(len(datatucker['energy(kWh)']))*baseline['energy(kWh)'].to_numpy())*100
datatucker.loc[:,'per_energy(kWh)']=pd.to_numeric(datatucker.loc[:,'per_energy(kWh)'],errors='coerce')
datatucker.loc[:,'per_energy(kWh)']=np.around(datatucker.loc[:,'per_energy(kWh)'], decimals=2)




datatt.loc[:, 'energy(kWh)'] = pd.to_numeric(datatt['energy(kWh)'], errors='coerce')
datatt['delta_energy(kWh)']=np.ones(len(datatt['energy(kWh)']))*baseline['energy(kWh)'].to_numpy()-datatt['energy(kWh)']
datatt.loc[:,'delta_energy(kWh)']=pd.to_numeric(datatt['delta_energy(kWh)'], errors='coerce')
datatt['per_energy(kWh)']=datatt['delta_energy(kWh)']/(np.ones(len(datatt['energy(kWh)']))*baseline['energy(kWh)'].to_numpy())*100
datatt.loc[:,'per_energy(kWh)']=pd.to_numeric(datatt.loc[:,'per_energy(kWh)'],errors='coerce')
datatt.loc[:,'per_energy(kWh)']=np.around(datatt.loc[:,'per_energy(kWh)'], decimals=2)

datatt['MACred']=(datatt['MAC_original']-datatt['MAC'])
datatt.loc[:,'MACred']=pd.to_numeric(datatt.loc[:,'MACred'],errors='coerce')
datatt.loc[:,'MACred']=np.around(datatt.loc[:,'MACred'], decimals=2)

datacp['MACred']=(datacp['MAC_original']-datacp['MAC'])
datacp.loc[:,'MACred']=pd.to_numeric(datacp.loc[:,'MACred'],errors='coerce')
datacp.loc[:,'MACred']=np.around(datacp.loc[:,'MACred'], decimals=2)


datatucker['MACred']=(datatucker['MAC_original']-datatucker['MAC'])
datatucker.loc[:,'MACred']=pd.to_numeric(datatucker.loc[:,'MACred'],errors='coerce')
datatucker.loc[:,'MACred']=np.around(datatucker.loc[:,'MACred'], decimals=2)


heatmap_data_tt = datatt.pivot( "Comp","Layer", "MACred").sort_index(ascending=False)
heatmap_data_tuck = datatucker.pivot( "Comp","Layer", "MACred").sort_index(ascending=False)
heatmap_data_cp = datacp.pivot( "Comp","Layer", "MACred").sort_index(ascending=False)

# Concatenate all data
all_data = pd.concat([datacp, datatt, datatucker])

# Calculate vmin and vmax
vmin = all_data['MACred'].min()
vmax = all_data['MACred'].max()


# Create the heatmap for Tucker Decomposition
plt.figure(figsize=(5, 5))
sns.heatmap(heatmap_data_tuck, annot=False, fmt=".2e", cmap=cmap, cbar_kws={'label': 'MAC operations'}, vmin=vmin, vmax=vmax, annot_kws={"size": 12})
plt.title('Reduced number of MAC operations Tucker', pad=20)
plt.xlabel('Layer')  
plt.ylabel('Compression Ratio')
plt.show()

# Create the heatmap for CP Decomposition
plt.figure(figsize=(5, 5))
sns.heatmap(heatmap_data_cp, annot=False, fmt=".2e", cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={'label': 'Added memory (# params)'}, annot_kws={"size": 12})
plt.title('Reduced number of MAC operations CP', pad=20)
plt.xlabel('Layer')  
plt.ylabel('Compression Ratio')
plt.show()

# Create the heatmap for TT Decomposition
plt.figure(figsize=(5, 5))
sns.heatmap(heatmap_data_tt, annot=False, fmt=".2e", cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={'label': 'Added memory (# params)'}, annot_kws={"size": 12})
plt.title('Reduced number of MAC operations TT', pad=20)
plt.xlabel('Layer')  
plt.ylabel('Compression Ratio')
plt.show()










#%% Heatmaps for the differnt methods
# Convert the 'energy(kWh)' column to float, errors='coerce' will convert non-convertible values to NaN

sns.set(font_scale=1.1) 
cmap = sns.diverging_palette(240, 10, s=85, l=50, n=7, center="light", as_cmap=True)

# Define a custom diverging colormap with more emphasis around zero and softer colors
colors = [(0.0, "#5D001E"), (0.4, "#9B3156"), (0.48, "#F2C1C6"), (0.5, "#FFFFFF"), (0.52, "#D0F2EC"), (0.6, "#4AAFA9"), (1.0, "#004D47")]
colors = [(0.0, "#006400"), (0.2, "#228B22"), (0.4, "#ADFF2F"), 
          (0.5, "white"), 
          (0.6, "#FFC0CB"), (0.8, "#FF6347"), (1.0, "#8B0000")]
n_bins = 100  # Discretizes the interpolation into bins
cmap_name = 'custom_diverging'
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

colors = [(0.0, "#2066a8"), (0.2, "#3594cc"), (0.4, "#8cc5e3"), 
          (0.5, "white"), 
          (0.6, "#d8a6a6"), (0.8, "#c46666"), (1.0, "#a00000")]

colors = [(0.0,  "#a00000"), (0.2, "#c46666"), (0.4, "#d8a6a6"), 
          (0.5, "white"), 
          (0.6, "#8cc5e3"), (0.8, "#3594cc"), (1.0,  "#2066a8")]
# colors = [(0.0, "#00441b"), (0.2, "#006d2c"), (0.4, "#238b45"), 
#           (0.5, "white"), 
#           (0.6, "#fcae91"), (0.8, "#fb6a4a"), (1.0, "#a50f15")]
n_bins = 100  # Discretizes the interpolation into bins
cmap_name = 'custom_diverging'
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


plt.close('all')
datacp.loc[:, 'energy(kWh)'] = pd.to_numeric(datacp['energy(kWh)'], errors='coerce')
datacp['delta_energy(kWh)']=np.ones(len(datacp['energy(kWh)']))*baseline['energy(kWh)'].to_numpy()-datacp['energy(kWh)']
datacp.loc[:,'delta_energy(kWh)']=pd.to_numeric(datacp['delta_energy(kWh)'], errors='coerce')

datacp['per_energy(kWh)']=datacp['delta_energy(kWh)']/(np.ones(len(datacp['energy(kWh)']))*baseline['energy(kWh)'].to_numpy())*100
datacp.loc[:,'per_energy(kWh)']=pd.to_numeric(datacp.loc[:,'per_energy(kWh)'],errors='coerce')
datacp.loc[:,'per_energy(kWh)']=np.around(datacp.loc[:,'per_energy(kWh)'], decimals=2)



heatmap_data_cp = datacp.pivot( "Comp","Layer", "per_energy(kWh)")

datatucker.loc[:, 'energy(kWh)'] = pd.to_numeric(datatucker['energy(kWh)'], errors='coerce')
datatucker['delta_energy(kWh)']=np.ones(len(datatucker['energy(kWh)']))*baseline['energy(kWh)'].to_numpy()-datatucker['energy(kWh)']
datatucker.loc[:,'delta_energy(kWh)']=pd.to_numeric(datatucker['delta_energy(kWh)'], errors='coerce')
datatucker['per_energy(kWh)']=datatucker['delta_energy(kWh)']/(np.ones(len(datatucker['energy(kWh)']))*baseline['energy(kWh)'].to_numpy())*100
datatucker.loc[:,'per_energy(kWh)']=pd.to_numeric(datatucker.loc[:,'per_energy(kWh)'],errors='coerce')
datatucker.loc[:,'per_energy(kWh)']=np.around(datatucker.loc[:,'per_energy(kWh)'], decimals=2)


heatmap_data_tuck = datatucker.pivot( "Comp","Layer", "per_energy(kWh)")


datatt.loc[:, 'energy(kWh)'] = pd.to_numeric(datatt['energy(kWh)'], errors='coerce')
datatt['delta_energy(kWh)']=np.ones(len(datatt['energy(kWh)']))*baseline['energy(kWh)'].to_numpy()-datatt['energy(kWh)']
datatt.loc[:,'delta_energy(kWh)']=pd.to_numeric(datatt['delta_energy(kWh)'], errors='coerce')
datatt['per_energy(kWh)']=datatt['delta_energy(kWh)']/(np.ones(len(datatt['energy(kWh)']))*baseline['energy(kWh)'].to_numpy())*100
datatt.loc[:,'per_energy(kWh)']=pd.to_numeric(datatt.loc[:,'per_energy(kWh)'],errors='coerce')
datatt.loc[:,'per_energy(kWh)']=np.around(datatt.loc[:,'per_energy(kWh)'], decimals=2)


heatmap_data_tt = datatt.pivot( "Comp","Layer", "per_energy(kWh)")

# Concatenate all data
all_data = pd.concat([datacp, datatt, datatucker])

# Calculate vmin and vmax
vmin = all_data['per_energy(kWh)'].min()
vmax = np.abs(vmin
)

# Creating the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data_tuck, annot=True, fmt=".2f", cmap=cmap,cbar_kws={'label': 'Energy Savings (%)'}, vmin=vmin,vmax=vmax,annot_kws={"size": 12})
plt.title('Percentage Energy Savings by Layer and Compression Ratio Using Tucker Decomposition', pad=20)
plt.xlabel('Layer')  
plt.ylabel('Compression Ratio')

# Creating the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data_cp, annot=True, fmt=".2f", cmap=cmap, vmin=vmin,vmax=vmax,cbar_kws={'label': 'Energy Savings (%)'},annot_kws={"size": 12})
sns.diverging_palette(145, 300, s=60, as_cmap=True)
plt.title('Percentage Energy Savings by Layer and Compression Ratio Using CP Decomposition', pad=20)
plt.xlabel('Layer')  
plt.ylabel('Compression Ratio')
plt.show()

# Creating the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data_tt
            , annot=True, fmt=".2f", cmap=cmap, vmin=vmin,vmax=vmax,cbar_kws={'label': 'Energy Savings (%)'},annot_kws={"size": 12})
plt.title('Percentage Energy Savings by Layer and Compression Ratio Using TT Decomposition', pad=20)
plt.xlabel('Layer')  
plt.ylabel('Compression Ratio')



#%%Total energy savings for the three methods
plt.close('all')
compression=datacp['Comp'].to_numpy()
tot_cp_eng=datacp['energy(kWh)'].to_numpy()
tot_tt_eng=datatt['energy(kWh)'].to_numpy()
tot_tuck_eng=datatucker['energy(kWh)'].to_numpy()

baseline_eng=baseline['energy(kWh)'].to_numpy()
baseline_eng=baseline_eng*np.ones((len(tot_cp_eng)))

delta_eng_cp=baseline_eng-tot_cp_eng
delta_eng_tt=baseline_eng-tot_tt_eng
delta_eng_tuck=baseline_eng-tot_tuck_eng

plt.figure()
plt.scatter(compression,delta_eng_cp, label='CP')
plt.title('Energy saving for CP for all layers')
plt.xlabel('Compression')
plt.ylabel('Energy saved (kWh)')
plt.legend()
plt.figure()
plt.scatter(compression,delta_eng_tt, label='TT')
plt.title('Energy saving for TT for all layers')
plt.xlabel('Compression')
plt.ylabel('Energy saved (kWh)')
plt.legend()
plt.figure()
plt.scatter(compression,delta_eng_tuck, label='Tucker')
plt.title('Energy saving for Tucker for all layers')
plt.xlabel('Compression')
plt.ylabel('Energy saved (kWh)')
plt.legend()


#%%MACs plots
plt.close('all')
datacp_str2=datacp[datacp['Stride'].isin([2])]
MACS=datacp_str2['MAC'].to_numpy()
MACS_org=datacp_str2['MAC_original'].to_numpy()
ratio_mac_str2=(MACS_org-MACS)/MACS_org*100
compr_str2=datacp_str2['Comp'].to_numpy()

datacp_str1=datacp[datacp['Stride'].isin([1])]
MACS=datacp_str1['MAC'].to_numpy()
MACS_org=datacp_str1['MAC_original'].to_numpy()
ratio_mac_str1=(MACS_org-MACS)/MACS_org*100
compr_str1=datacp_str1['Comp'].to_numpy()

plt.figure()
plt.scatter(compr_str2,ratio_mac_str2,label='Stride=2')
plt.scatter(compr_str1,ratio_mac_str1,label='Stride=1')
plt.legend()
plt.ylabel('Reduction ratio MACs')
plt.xlabel('Compression ratio')
plt.title('MAC reduction (%) using CP decomposition')
plt.ylim([-25,100])

datatt_str2=datatt[datatt['Stride'].isin([2])]
MACS=datatt_str2['MAC'].to_numpy()
MACS_org=datatt_str2['MAC_original'].to_numpy()
ratio_mac_str2=(MACS_org-MACS)/MACS_org*100
compr_str2=datatt_str2['Comp'].to_numpy()

datatt_str1=datatt[datatt['Stride'].isin([1])]
MACS=datatt_str1['MAC'].to_numpy()
MACS_org=datatt_str1['MAC_original'].to_numpy()
ratio_mac_str1=(MACS_org-MACS)/MACS_org*100
compr_str1=datatt_str1['Comp'].to_numpy()

plt.figure()
plt.scatter(compr_str2,ratio_mac_str2,label='Stride=2')
plt.scatter(compr_str1,ratio_mac_str1,label='Stride=1')
plt.legend()
plt.ylabel('Reduction ratio MACs')
plt.xlabel('Compression ratio')
plt.title('MAC reduction (%) using TT decomposition')
plt.ylim([-25,100])


datatucker_str2=datatucker[datatucker['Stride'].isin([2])]
MACS=datatucker_str2['MAC'].to_numpy()
MACS_org=datatucker_str2['MAC_original'].to_numpy()
ratio_mac_str2=(MACS_org-MACS)/MACS_org*100
compr_str2=datatucker_str2['Comp'].to_numpy()

datatucker_str1=datatucker[datatucker['Stride'].isin([1])]
MACS=datatucker_str1['MAC'].to_numpy()
MACS_org=datatucker_str1['MAC_original'].to_numpy()
ratio_mac_str1=(MACS_org-MACS)/MACS_org*100
compr_str1=datatucker_str1['Comp'].to_numpy()

plt.figure()
plt.scatter(compr_str2,ratio_mac_str2,label='Stride=2')
plt.scatter(compr_str1,ratio_mac_str1,label='Stride=1')
plt.legend()
plt.ylabel('Reduction MAC operations [%]')
plt.xlabel('Compression ratio')
plt.title('MAC reduction (%) using Tucker decomposition')
plt.ylim([-25,100])


#%%
plt.close('all')

# Define colors for each method
colors = {'CP': '#EE6677', 'TT': '#228833', 'Tucker': '#66CCEE'}
# Define unique markers for different measurements
markers = ['o', 's', '^', 'D', 'v', 'p', '*', '+', 'x', '<', '>']

# Custom legend handles
cp_handle = mlines.Line2D([], [], color=colors['CP'], marker='o', linestyle='None', markersize=10, label='CP decomposition')
tt_handle = mlines.Line2D([], [], color=colors['TT'], marker='*', linestyle='None', markersize=10, label='TT decomposition')
tucker_handle = mlines.Line2D([], [], color=colors['Tucker'], marker='D', linestyle='None', markersize=10, label='Tucker decomposition')

# Marker size and transparency
marker_size = 60
alpha = 0.3

# Stride 1
datacp_str1 = datacp[datacp['Stride'].isin([1])]
MACS_cp_str1 = datacp_str1['MAC'].to_numpy()
MACS_cp_org_str1 = datacp_str1['MAC_original'].to_numpy()
ratio_mac_cp_str1 = (MACS_cp_org_str1 - MACS_cp_str1) / MACS_cp_org_str1 * 100
compr_cp_str1 = datacp_str1['Comp'].to_numpy()

datatt_str1 = datatt[datatt['Stride'].isin([1])]
MACS_tt_str1 = datatt_str1['MAC'].to_numpy()
MACS_tt_org_str1 = datatt_str1['MAC_original'].to_numpy()
ratio_mac_tt_str1 = (MACS_tt_org_str1 - MACS_tt_str1) / MACS_tt_org_str1 * 100
compr_tt_str1 = datatt_str1['Comp'].to_numpy()

datatucker_str1 = datatucker[datatucker['Stride'].isin([1])]
MACS_tucker_str1 = datatucker_str1['MAC'].to_numpy()
MACS_tucker_org_str1 = datatucker_str1['MAC_original'].to_numpy()
ratio_mac_tucker_str1 = (MACS_tucker_org_str1 - MACS_tucker_str1) / MACS_tucker_org_str1 * 100
compr_tucker_str1 = datatucker_str1['Comp'].to_numpy()

plt.figure()
ax = plt.gca()
ax.set_axisbelow(True)  # Ensure grid is drawn below the scatter plots
ax.grid(True, zorder=0)
plt.scatter(compr_cp_str1, ratio_mac_cp_str1, color=colors['CP'], marker='o', s=marker_size, alpha=alpha)
plt.scatter(compr_tt_str1, ratio_mac_tt_str1, color=colors['TT'], marker='*', s=marker_size, alpha=alpha)
plt.scatter(compr_tucker_str1, ratio_mac_tucker_str1, color=colors['Tucker'], marker="D", s=marker_size, alpha=alpha)

plt.legend(handles=[cp_handle, tt_handle, tucker_handle], loc='lower left')
plt.ylabel('Reduction ratio MACs [%]')
plt.xlabel('Compression ratio')
plt.title('MAC reduction (%) for Stride 1')
plt.ylim([-25, 100])

# Stride 2
datacp_str2 = datacp[datacp['Stride'].isin([2])]
MACS_cp_str2 = datacp_str2['MAC'].to_numpy()
MACS_cp_org_str2 = datacp_str2['MAC_original'].to_numpy()
ratio_mac_cp_str2 = (MACS_cp_org_str2 - MACS_cp_str2) / MACS_cp_org_str2 * 100
compr_cp_str2 = datacp_str2['Comp'].to_numpy()

datatt_str2 = datatt[datatt['Stride'].isin([2])]
MACS_tt_str2 = datatt_str2['MAC'].to_numpy()
MACS_tt_org_str2 = datatt_str2['MAC_original'].to_numpy()
ratio_mac_tt_str2 = (MACS_tt_org_str2 - MACS_tt_str2) / MACS_tt_org_str2 * 100
compr_tt_str2 = datatt_str2['Comp'].to_numpy()

datatucker_str2 = datatucker[datatucker['Stride'].isin([2])]
MACS_tucker_str2 = datatucker_str2['MAC'].to_numpy()
MACS_tucker_org_str2 = datatucker_str2['MAC_original'].to_numpy()
ratio_mac_tucker_str2 = (MACS_tucker_org_str2 - MACS_tucker_str2) / MACS_tucker_org_str2 * 100
compr_tucker_str2 = datatucker_str2['Comp'].to_numpy()

plt.figure()
ax = plt.gca()
ax.set_axisbelow(True)  # Ensure grid is drawn below the scatter plots
ax.grid(True, zorder=0)
plt.scatter(compr_cp_str2, ratio_mac_cp_str2, color=colors['CP'], marker='o', s=marker_size, alpha=alpha)
plt.scatter(compr_tt_str2, ratio_mac_tt_str2, color=colors['TT'], marker='*', s=marker_size, alpha=alpha)
plt.scatter(compr_tucker_str2, ratio_mac_tucker_str2, color=colors['Tucker'], marker='D', s=marker_size, alpha=alpha)

plt.legend(handles=[cp_handle, tt_handle, tucker_handle], loc='lower left')
plt.ylabel('Reduction ratio MACs [%]')
plt.xlabel('Compression ratio')
plt.title('MAC reduction (%) for Stride 2')
plt.ylim([40, 100])

plt.show()
ax = plt.gca()
ax.set_axisbelow(True)  # Ensure grid is drawn below the scatter plots
ax.grid(True, zorder=0)#%%%Energy Conv blocks layers

#%%
vmin = all_data['per_energy(kWh)'].min()-2
vmax = all_data['per_energy(kWh)'].max()+2

datatucker['per_energy_std(kWh)']=100*np.sqrt((datatucker['std_energy(kWh)'].to_numpy(dtype=float)/baseline['energy(kWh)'].to_numpy(dtype=float))**2+((np.multiply(-datatucker['delta_energy(kWh)'].to_numpy(dtype=float), datatucker['std_energy(kWh)'].to_numpy(dtype=float)))/datatucker['energy(kWh)'].to_numpy(dtype=float))**2)
datatucker.loc[:,'per_energy_std(kWh)']=pd.to_numeric(datatucker.loc[:,'per_energy_std(kWh)'],errors='coerce')


datacp['per_energy_std(kWh)']=100*np.sqrt((datacp['std_energy(kWh)'].to_numpy(dtype=float)/baseline['energy(kWh)'].to_numpy(dtype=float))**2+((np.multiply(-datacp['delta_energy(kWh)'].to_numpy(dtype=float), datacp['std_energy(kWh)'].to_numpy(dtype=float)))/datacp['energy(kWh)'].to_numpy(dtype=float))**2)
datacp.loc[:,'per_energy_std(kWh)']=pd.to_numeric(datacp.loc[:,'per_energy_std(kWh)'],errors='coerce')


datatt['per_energy_std(kWh)']=100*np.sqrt((datatt['std_energy(kWh)'].to_numpy(dtype=float)/baseline['energy(kWh)'].to_numpy(dtype=float))**2+((np.multiply(-datatt['delta_energy(kWh)'].to_numpy(dtype=float), datatt['std_energy(kWh)'].to_numpy(dtype=float)))/datatt['energy(kWh)'].to_numpy(dtype=float))**2)
datatt.loc[:,'per_energy_std(kWh)']=pd.to_numeric(datatt.loc[:,'per_energy_std(kWh)'],errors='coerce')

plt.close('all')


lay=[51,54,57,60,63]
color_codes =  [(214/255, 39/255, 40/255), (44/255, 160/255, 44/255), (31/255, 119/255, 180/255), (255/255, 127/255, 14/255), (148/255, 103/255, 189/255)]
color = get_color()

for i in ['cp', 'tucker', 'tt']:
    plt.figure()
    color=get_color()
    for layer in lay:
        if i == 'cp':
            datacurr = datacp.copy()
        elif i == 'tt':
            datacurr = datatt.copy()
        elif i == 'tucker':
            datacurr = datatucker.copy()
        
        data_cp_down = datacurr[datacurr['Layer'].isin([layer])]
        delta_eng = data_cp_down['per_energy(kWh)']
        std = data_cp_down['per_energy_std(kWh)']
        compr = data_cp_down['Comp'].to_numpy()
        
        z = np.polyfit(compr, delta_eng, 1)
        p = np.poly1d(z)
        xp = np.linspace(0, 0.9, 100)
        
        acolor = next(color)
        plt.errorbar(compr, delta_eng, yerr=std, fmt="o", color=acolor)
        plt.scatter(compr, delta_eng, color=acolor, marker='o', label=f'Layer {layer}')
        plt.plot(xp, p(xp), acolor + '-')
        
        
    plt.legend()
    plt.ylabel('Energy (%)')
    plt.xlabel('Compression')
    plt.title(f'Percentage of energy saved ResNet18 Basic Block 4 With {i}')

plt.show()

lay=[6,15]
for i in ['cp', 'tucker', 'tt']:
    plt.figure()
    color=get_color()
    for layer in lay:
        if i=='cp':
            datacurr=datacp.copy()
        elif i=='tt':
            datacurr=datatt.copy()
        elif i=='tucker':
            datacurr=datatucker.copy()
        data_cp_down=datacurr[datacurr['Layer'].isin([layer])]
        delta_eng=data_cp_down['per_energy(kWh)']
        std=data_cp_down['per_energy_std(kWh)']
        compr= data_cp_down['Comp'].to_numpy()
        
        
        z = np.polyfit(compr, delta_eng, 1)
        p = np.poly1d(z)
        xp = np.linspace(0, 0.9, 100)
        
        acolor = next(color)
        plt.errorbar(compr, delta_eng, yerr=std, fmt="o", color=acolor)
        plt.scatter(compr, delta_eng, color=acolor, marker='o', label=f'Layer {layer}')
        plt.plot(xp, p(xp), acolor + '-')

    plt.legend()
    plt.ylim([-15,26])
    plt.ylim([-35,9])
    plt.ylabel('Energy (%)')
    plt.xlabel('Compression')
    plt.title(f'Percentage of energy saved ResNet18 Basic Block 1 With {i}')
    
lay=[19,22,25,28]
for i in ['cp', 'tucker', 'tt']:
    plt.figure()
    color=get_color()
    for layer in lay:
        if i=='cp':
            datacurr=datacp.copy()
        elif i=='tt':
            datacurr=datatt.copy()
        elif i=='tucker':
            datacurr=datatucker.copy()
        data_cp_down=datacurr[datacurr['Layer'].isin([layer])]
        data_cp_down=datacurr[datacurr['Layer'].isin([layer])]
        delta_eng=data_cp_down['per_energy(kWh)']
        std=data_cp_down['per_energy_std(kWh)']
        
        z = np.polyfit(compr, delta_eng, 1)
        p = np.poly1d(z)
        xp = np.linspace(0, 0.9, 100)
        
        acolor = next(color)
        plt.errorbar(compr, delta_eng, yerr=std, fmt="o", color=acolor)
        plt.scatter(compr, delta_eng, color=acolor, marker='o', label=f'Layer {layer}')
        plt.plot(xp, p(xp), acolor + '-')

    plt.legend()
    plt.ylim([-25,5])
    plt.ylabel('Energy (%)')
    plt.xlabel('Compression')
    plt.title(f'Percentage of energy saved ResNet18 Basic Block 2 With {i}')

lay=[35,38,41,47]
for i in ['cp', 'tucker', 'tt']:
    plt.figure()
    color=get_color()
    for layer in lay:
        if i=='cp':
            datacurr=datacp.copy()
        elif i=='tt':
            datacurr=datatt.copy()
        elif i=='tucker':
            datacurr=datatucker.copy()
        data_cp_down=datacurr[datacurr['Layer'].isin([layer])]
        delta_eng=data_cp_down['per_energy(kWh)']
        std=data_cp_down['per_energy_std(kWh)']
        compr= data_cp_down['Comp'].to_numpy()
        
        z = np.polyfit(compr, delta_eng, 1)
        p = np.poly1d(z)
        xp = np.linspace(0, 0.9, 100)
        
        acolor = next(color)
        plt.errorbar(compr, delta_eng, yerr=std, fmt="o", color=acolor)
        plt.scatter(compr, delta_eng, color=acolor, marker='o', label=f'Layer {layer}')
        plt.plot(xp, p(xp), acolor + '-')

    plt.legend()
    plt.ylim([-13,7])
    plt.ylabel('Energy (%)')
    plt.xlabel('Compression')
    plt.title(f'Percentage of energy saved ResNet18 Basic Block 3 With {i}')

#%%%MACS Conv blocks layers
plt.close('all')
lay=[51,54,57,60,63]
for i in ['cp', 'tucker', 'tt']:
    plt.figure()
    color=get_color()
    for layer in lay:
        if i=='cp':
            datacurr=datacp.copy()
        elif i=='tt':
            datacurr=datatt.copy()
        elif i=='tucker':
            datacurr=datatucker.copy()
        data_cp_down=datacurr[datacurr['Layer'].isin([layer])]
        delta_eng=data_cp_down['per_energy(kWh)']
        std=data_cp_down['per_energy_std(kWh)']
        compr= data_cp_down['MAC'].to_numpy()
        
        
        z = np.polyfit(compr, delta_eng, 2)
        p = np.poly1d(z)
        xp = np.linspace(0, 1, 100)
        
        acolor = next(color)
        plt.errorbar(compr, delta_eng, yerr=std, fmt="o", color=acolor)
        plt.scatter(compr, delta_eng, color=acolor, marker='o', label=f'Layer {layer}')
        plt.plot(xp, p(xp), acolor + '-')
        
    plt.legend()
    #plt.ylim((vmin,vmax))
    plt.title(f'ResNet18 Basic Block 4 With {i}')


lay=[6,15]
for i in ['cp', 'tucker', 'tt']:
    plt.figure()
    color=get_color()
    for layer in lay:
        if i=='cp':
            datacurr=datacp.copy()
        elif i=='tt':
            datacurr=datatt.copy()
        elif i=='tucker':
            datacurr=datatucker.copy()
        data_cp_down=datacurr[datacurr['Layer'].isin([layer])]
        delta_eng=data_cp_down['per_energy(kWh)']
        std=data_cp_down['per_energy_std(kWh)']
        compr= data_cp_down['MAC'].to_numpy()
        
        
        z = np.polyfit(compr, delta_eng, 2)
        p = np.poly1d(z)
        xp = np.linspace(0, 1, 100)
        
        acolor = next(color)
        plt.errorbar(compr, delta_eng, yerr=std, fmt="o", color=acolor)
        plt.scatter(compr, delta_eng, color=acolor, marker='o', label=f'Layer {layer}')
        plt.plot(xp, p(xp), acolor + '-')

    plt.legend()
    #plt.ylim((vmin,vmax))
    plt.title(f'ResNet18 Basic Block 1 with {i}')
    
lay=[19,22,25,28]
for i in ['cp', 'tucker', 'tt']:
    plt.figure()
    color=get_color()
    for layer in lay:
        if i=='cp':
            datacurr=datacp.copy()
        elif i=='tt':
            datacurr=datatt.copy()
        elif i=='tucker':
            datacurr=datatucker.copy()
        data_cp_down=datacurr[datacurr['Layer'].isin([layer])]
        delta_eng=data_cp_down['per_energy(kWh)']
        std=data_cp_down['per_energy_std(kWh)']
        compr= data_cp_down['MAC'].to_numpy()
        
        
        z = np.polyfit(compr, delta_eng, 2)
        p = np.poly1d(z)
        xp = np.linspace(0, 1, 100)
        
        acolor = next(color)
        plt.errorbar(compr, delta_eng, yerr=std, fmt="o", color=acolor)
        plt.scatter(compr, delta_eng, color=acolor, marker='o', label=f'Layer {layer}')
        plt.plot(xp, p(xp), acolor + '-')

    plt.legend()
    #plt.ylim((vmin,vmax))
    plt.title(f'ResNet18 Basic Block 2 with {i}')

lay=[35,38,41,47]
for i in ['cp', 'tucker', 'tt']:
    plt.figure()
    color=get_color()
    for layer in lay:
        if i=='cp':
            datacurr=datacp.copy()
        elif i=='tt':
            datacurr=datatt.copy()
        elif i=='tucker':
            datacurr=datatucker.copy()
        data_cp_down=datacurr[datacurr['Layer'].isin([layer])]
        delta_eng=data_cp_down['per_energy(kWh)']
        std=data_cp_down['per_energy_std(kWh)']
        compr= data_cp_down['MAC'].to_numpy()
        
        
        z = np.polyfit(compr, delta_eng, 2)
        p = np.poly1d(z)
        xp = np.linspace(0, 1, 100)
        
        acolor = next(color)
        plt.errorbar(compr, delta_eng, yerr=std, fmt="o", color=acolor)
        plt.scatter(compr, delta_eng, color=acolor, marker='o', label=f'Layer {layer}')
        plt.plot(xp, p(xp), acolor + '-')

    plt.legend()
    #plt.ylim((vmin,vmax))
    plt.title(f'ResNet18 Basic Block 3 with {i}')       
    
#%%%MACS_ratio Conv blocks layers
plt.close('all')
lay=[51,54,57,60,63]
for i in ['cp', 'tucker', 'tt']:
    plt.figure()
    color=get_color()
    for layer in lay:
        if i=='cp':
            datacurr=datacp.copy()
        elif i=='tt':
            datacurr=datatt.copy()
        elif i=='tucker':
            datacurr=datatucker.copy()
        data_cp_down=datacurr[datacurr['Layer'].isin([layer])]
        delta_eng=data_cp_down['per_energy(kWh)']
        std=data_cp_down['per_energy_std(kWh)']
        compr= data_cp_down['MAC'].to_numpy()/data_cp_down['MAC_original'].to_numpy()
        
        
        z = np.polyfit(compr, delta_eng, 2)
        p = np.poly1d(z)
        xp = np.linspace(0, 1, 100)
        
        acolor = next(color)
        plt.errorbar(compr, delta_eng, yerr=std, fmt="o", color=acolor)
        plt.scatter(compr, delta_eng, color=acolor, marker='o', label=f'Layer {layer}')
        plt.plot(xp, p(xp), acolor + '-')
    plt.legend()
    #plt.ylim((vmin,vmax))
    plt.title(f'ResNet18 Basic Block 4 With {i}')


lay=[6,15]
for i in ['cp', 'tucker', 'tt']:
    plt.figure()
    color=get_color()
    for layer in lay:
        if i=='cp':
            datacurr=datacp.copy()
        elif i=='tt':
            datacurr=datatt.copy()
        elif i=='tucker':
            datacurr=datatucker.copy()
        data_cp_down=datacurr[datacurr['Layer'].isin([layer])]
        delta_eng=data_cp_down['per_energy(kWh)']
        std=data_cp_down['per_energy_std(kWh)']
        compr= data_cp_down['MAC'].to_numpy()/data_cp_down['MAC_original'].to_numpy()
        
        
        z = np.polyfit(compr, delta_eng, 2)
        p = np.poly1d(z)
        xp = np.linspace(0, 1, 100)
        
        acolor = next(color)
        plt.errorbar(compr, delta_eng, yerr=std, fmt="o", color=acolor)
        plt.scatter(compr, delta_eng, color=acolor, marker='o', label=f'Layer {layer}')
        plt.plot(xp, p(xp), acolor + '-')

    plt.legend()
    #plt.ylim((vmin,vmax))
    plt.title(f'ResNet18 Basic Block 1 with {i}')
    
lay=[19,22,25,28]
for i in ['cp', 'tucker', 'tt']:
    plt.figure()
    color=get_color()
    for layer in lay:
        if i=='cp':
            datacurr=datacp.copy()
        elif i=='tt':
            datacurr=datatt.copy()
        elif i=='tucker':
            datacurr=datatucker.copy()
        data_cp_down=datacurr[datacurr['Layer'].isin([layer])]
        delta_eng=data_cp_down['per_energy(kWh)']
        std=data_cp_down['per_energy_std(kWh)']
        compr= data_cp_down['MAC'].to_numpy()/data_cp_down['MAC_original'].to_numpy()
        
        
        z = np.polyfit(compr, delta_eng, 2)
        p = np.poly1d(z)
        xp = np.linspace(0, 1, 100)
        
        acolor = next(color)
        plt.errorbar(compr, delta_eng, yerr=std, fmt="o", color=acolor)
        plt.scatter(compr, delta_eng, color=acolor, marker='o', label=f'Layer {layer}')
        plt.plot(xp, p(xp), acolor + '-')

    plt.legend()
    #plt.ylim((vmin,vmax))
    plt.title(f'ResNet18 Basic Block 2 with {i}')

lay=[35,38,41,47]
for i in ['cp', 'tucker', 'tt']:
    plt.figure()
    color=get_color()
    for layer in lay:
        if i=='cp':
            datacurr=datacp.copy()
        elif i=='tt':
            datacurr=datatt.copy()
        elif i=='tucker':
            datacurr=datatucker.copy()
        data_cp_down=datacurr[datacurr['Layer'].isin([layer])]
        delta_eng=data_cp_down['per_energy(kWh)']
        std=data_cp_down['per_energy_std(kWh)']
        compr= data_cp_down['MAC'].to_numpy()/data_cp_down['MAC_original'].to_numpy()
        
        z = np.polyfit(compr, delta_eng, 2)
        p = np.poly1d(z)
        xp = np.linspace(0, 1, 100)
        
        acolor = next(color)
        plt.errorbar(compr, delta_eng, yerr=std, fmt="o", color=acolor)
        plt.scatter(compr, delta_eng, color=acolor, marker='o', label=f'Layer {layer}')
        plt.plot(xp, p(xp), acolor + '-')

    plt.legend()
    #plt.ylim((vmin,vmax))
    plt.title(f'ResNet18 Basic Block 3 with {i}')       
    
    
#%%The first layers

lay=[51,35,19,6]
for i in ['cp', 'tucker', 'tt']:
    plt.figure()
    color=get_color()
    for layer in lay:
        if i=='cp':
            datacurr=datacp.copy()
        elif i=='tt':
            datacurr=datatt.copy()
        elif i=='tucker':
            datacurr=datatucker.copy()
        data_cp_down=datacurr[datacurr['Layer'].isin([layer])]
        delta_eng=data_cp_down['per_energy(kWh)']
        std=data_cp_down['per_energy_std(kWh)']
        compr= data_cp_down['Comp'].to_numpy()

        
        z = np.polyfit(compr, delta_eng, 2)
        p = np.poly1d(z)
        xp = np.linspace(0, 1, 100)
        
        acolor = next(color)
        plt.errorbar(compr, delta_eng, yerr=std, fmt="o", color=acolor)
        plt.scatter(compr, delta_eng, color=acolor, marker='o', label=f'Layer {layer}')
        plt.plot(xp, p(xp), acolor + '-')

    plt.legend()
    plt.ylim((vmin,vmax))
    plt.title(f'ResNet18 Basic Block 3 with {i}')  
    
#%% Heatmaps for the differnt methods for diferent deltas
# Convert the 'energy(kWh)' column to float, errors='coerce' will convert non-convertible values to NaN

sns.set(font_scale=1.1) 
cmap = sns.diverging_palette(240, 10, s=85, l=50, n=7, center="light", as_cmap=True)

# Define a custom diverging colormap with more emphasis around zero and softer colors
colors = [(0.0, "#5D001E"), (0.4, "#9B3156"), (0.48, "#F2C1C6"), (0.5, "#FFFFFF"), (0.52, "#D0F2EC"), (0.6, "#4AAFA9"), (1.0, "#004D47")]
colors = [(0.0, "#006400"), (0.2, "#228B22"), (0.4, "#ADFF2F"), 
          (0.5, "white"), 
          (0.6, "#FFC0CB"), (0.8, "#FF6347"), (1.0, "#8B0000")]
n_bins = 100  # Discretizes the interpolation into bins
cmap_name = 'custom_diverging'
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

colors = [(0.0, "#2066a8"), (0.2, "#3594cc"), (0.4, "#8cc5e3"), 
          (0.5, "white"), 
          (0.6, "#d8a6a6"), (0.8, "#c46666"), (1.0, "#a00000")]

colors = [(0.0,  "#a00000"), (0.2, "#c46666"), (0.4, "#d8a6a6"), 
          (0.5, "white"), 
          (0.6, "#8cc5e3"), (0.8, "#3594cc"), (1.0,  "#2066a8")]
# colors = [(0.0, "#00441b"), (0.2, "#006d2c"), (0.4, "#238b45"), 
#           (0.5, "white"), 
#           (0.6, "#fcae91"), (0.8, "#fb6a4a"), (1.0, "#a50f15")]
n_bins = 100  # Discretizes the interpolation into bins
cmap_name = 'custom_diverging'
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


plt.close('all')
datacp.loc[:, 'energy(kWh)'] = pd.to_numeric(datacp['energy(kWh)'], errors='coerce')
datacp['delta_energy(kWh)']=np.ones(len(datacp['energy(kWh)']))*baseline['energy(kWh)'].to_numpy()-datacp['energy(kWh)']
datacp.loc[:,'delta_energy(kWh)']=pd.to_numeric(datacp['delta_energy(kWh)'], errors='coerce')

datacp['per_energy(kWh)']=datacp['delta_energy(kWh)']/(np.ones(len(datacp['energy(kWh)']))*baseline['energy(kWh)'].to_numpy())*100
datacp.loc[:,'per_energy(kWh)']=pd.to_numeric(datacp.loc[:,'per_energy(kWh)'],errors='coerce')
datacp.loc[:,'per_energy(kWh)']=np.around(datacp.loc[:,'per_energy(kWh)'], decimals=2)

datacp['in_out']=datacp['In_ch']-datacp['Out_ch']



heatmap_data_cp = datacp.pivot( "Comp","Layer", "per_energy(kWh)")

datatucker.loc[:, 'energy(kWh)'] = pd.to_numeric(datatucker['energy(kWh)'], errors='coerce')
datatucker['delta_energy(kWh)']=np.ones(len(datatucker['energy(kWh)']))*baseline['energy(kWh)'].to_numpy()-datatucker['energy(kWh)']
datatucker.loc[:,'delta_energy(kWh)']=pd.to_numeric(datatucker['delta_energy(kWh)'], errors='coerce')
datatucker['per_energy(kWh)']=datatucker['delta_energy(kWh)']/(np.ones(len(datatucker['energy(kWh)']))*baseline['energy(kWh)'].to_numpy())*100
datatucker.loc[:,'per_energy(kWh)']=pd.to_numeric(datatucker.loc[:,'per_energy(kWh)'],errors='coerce')
datatucker.loc[:,'per_energy(kWh)']=np.around(datatucker.loc[:,'per_energy(kWh)'], decimals=2)

heatmap_data_tuck = datatucker.pivot( "Comp","Layer", "per_energy(kWh)")


datatt.loc[:, 'energy(kWh)'] = pd.to_numeric(datatt['energy(kWh)'], errors='coerce')
datatt['delta_energy(kWh)']=np.ones(len(datatt['energy(kWh)']))*baseline['energy(kWh)'].to_numpy()-datatt['energy(kWh)']
datatt.loc[:,'delta_energy(kWh)']=pd.to_numeric(datatt['delta_energy(kWh)'], errors='coerce')
datatt['per_energy(kWh)']=datatt['delta_energy(kWh)']/(np.ones(len(datatt['energy(kWh)']))*baseline['energy(kWh)'].to_numpy())*100
datatt.loc[:,'per_energy(kWh)']=pd.to_numeric(datatt.loc[:,'per_energy(kWh)'],errors='coerce')
datatt.loc[:,'per_energy(kWh)']=np.around(datatt.loc[:,'per_energy(kWh)'], decimals=2)

heatmap_data_tt = datatt.pivot( "Comp","Layer", "per_energy(kWh)")

# Concatenate all data
all_data = pd.concat([datacp, datatt, datatucker])

# Calculate vmin and vmax
vmin = all_data['per_energy(kWh)'].min()
vmax = np.abs(vmin)

# Creating the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data_tuck, annot=True, fmt=".2f", cmap=cmap,cbar_kws={'label': 'Energy Savings (%)'}, vmin=vmin,vmax=vmax,annot_kws={"size": 12})
plt.title('Percentage Energy Savings by Layer and Compression Ratio Using Tucker Decomposition', pad=20)
plt.xlabel('Layer')  
plt.ylabel('Compression Ratio')

# Creating the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data_cp, annot=True, fmt=".2f", cmap=cmap, vmin=vmin,vmax=vmax,cbar_kws={'label': 'Energy Savings (%)'},annot_kws={"size": 12})
sns.diverging_palette(145, 300, s=60, as_cmap=True)
plt.title('Percentage Energy Savings by Layer and Compression Ratio Using CP Decomposition', pad=20)
plt.xlabel('Layer')  
plt.ylabel('Compression Ratio')
plt.show()

# Creating the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data_tt
            , annot=True, fmt=".2f", cmap=cmap, vmin=vmin,vmax=vmax,cbar_kws={'label': 'Energy Savings (%)'},annot_kws={"size": 12})
plt.title('Percentage Energy Savings by Layer and Compression Ratio Using TT Decomposition', pad=20)
plt.xlabel('Layer')  
plt.ylabel('Compression Ratio')
