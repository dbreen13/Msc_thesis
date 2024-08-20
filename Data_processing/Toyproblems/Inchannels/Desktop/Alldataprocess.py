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
import svgwrite


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

def process_data(total_df_inch, data_bas, dec_method):
    # Filter data for the specific decomposition method
    data_tt = total_df_inch[total_df_inch['Dec'] == dec_method]
    
    # Initialize an empty list to store processed data
    processed_data = []

    # Define the input channels to process
    input_channels = [192, 256, 320, 384]
    
    for in_ch in input_channels:
        # Filter data for the specific input channel
        data_inch = data_tt[data_tt['In_ch'] == in_ch]
        data_bas_inch = data_bas[data_bas['In_ch'] == in_ch]
        
        # Convert energy(kWh) to numeric
        data_inch.loc[:, 'energy(kWh)'] = pd.to_numeric(data_inch['energy(kWh)'], errors='coerce')
        
        # Calculate delta energy
        
        data_inch['delta_energy(kWh)'] = np.ones(len(data_inch['energy(kWh)'])) * data_bas_inch['energy(kWh)'].to_numpy() - data_inch['energy(kWh)']
        data_inch.loc[:, 'delta_energy(kWh)'] = pd.to_numeric(data_inch['delta_energy(kWh)'], errors='coerce')
        
        # Calculate percentage energy
        data_inch['per_energy(kWh)'] = data_inch['delta_energy(kWh)'] / (np.ones(len(data_inch['energy(kWh)'])) * data_bas_inch['energy(kWh)'].to_numpy()) * 100
        data_inch.loc[:, 'per_energy(kWh)'] = pd.to_numeric(data_inch.loc[:, 'per_energy(kWh)'], errors='coerce')
        data_inch.loc[:, 'per_energy(kWh)'] = np.around(data_inch.loc[:, 'per_energy(kWh)'], decimals=2)
        
        # Append the processed data to the list
        processed_data.append(data_inch)
    
    # Concatenate all processed data
    processed_data_df = pd.concat(processed_data)
    
    return processed_data_df

def calculate_mac_reduction(data, compression_ratios):
    macs = {}
    for cr in compression_ratios:
        cr_data = data[data['Comp'] == cr]
        macs[cr] = cr_data['MAC_original'] - cr_data['MAC']
    return macs

def calculate_memory(data, compression_ratios, memory_type):
    memory = {}
    for cr in compression_ratios:
        cr_data = data[data['Comp'] == cr]
        memory[cr] = cr_data[memory_type]
    return memory

# Define plotting function
def plot_method(ax, method_data, method_name, colors, compression_ratios, input_channels, ylim=None):
    bar_width = 0.14
    x = np.arange(len(input_channels))
    for i, cr in enumerate(compression_ratios):
        offset = (i - 2) * (bar_width + 0.01)  # Adjust the offset for each bar within a group
        ax.bar(x + offset, method_data[cr], width=bar_width, color=colors[cr], label=f'{cr}', zorder=2)
    ax.set_title(method_name)
    ax.set_xticks(x)
    ax.set_xticklabels(input_channels)
    ax.grid(True, which='both', zorder=1)
    if ylim:
        ax.set_ylim(ylim)

#%%Combine all datasets
allinfo_inch=pd.read_pickle('allinfo_in.pkl')

data_tot=pd.DataFrame.from_dict(pd.read_pickle('data_tt_dec_inch.pkl')).transpose()

sort_keys=sorted(data_tot.index, key=custom_sort_key)
data_tot = data_tot.loc[sort_keys]



data_bas=pd.DataFrame.from_dict(pd.read_pickle('data_bas_inch.pkl')).transpose()
data_bas['In_ch'] = data_bas.index.to_series().str.extract(r'inch(\d+)-wh')[0].astype(float)

#Create the new index based on the specified format
new_index = [
    f"outch{row['Out_ch']}-inch{row['In_ch']}-fact{row['Dec']}-r{row['Comp']}-wh{row['In_feat'][2]}"
    for idx, row in allinfo_inch.iterrows()
]

# Assign the new index to the DataFrame
allinfo_inch.index = new_index


#load memory measurements
mem_inch=pd.read_pickle('mem_inch (1).pkl')
mem_inch=mem_inch.rename(columns={'Mem':'Mem_meas'})
mem_inch=mem_inch.loc[sort_keys]
mem_inch.index=new_index

mem_bas=pd.read_pickle('mem_bas_inch.pkl')

membas_tot=pd.concat([mem_bas] * 15, ignore_index=True)



#make final dataset
total_df_inch=pd.concat([data_tot,allinfo_inch,mem_inch], axis=1)
total_df_inch['Mem_meas_diff']=total_df_inch['Mem_meas']-membas_tot['Mem'].to_numpy()

mac_min=np.min((total_df_inch['MAC_original']-total_df_inch['MAC']))-9**8
mac_max=np.max((total_df_inch['MAC_original']-total_df_inch['MAC']))+9**8

save_path = "inchall_reg.pkl"
with open(save_path, 'wb') as f:
    pickle.dump(total_df_inch, f)
#%%
data_tucker_new = process_data(total_df_inch, data_bas, 'tucker')
data_cp_new = process_data(total_df_inch, data_bas, 'cp')
data_tt_new = process_data(total_df_inch, data_bas, 'tt')

datanew=pd.concat([data_cp_new, data_tucker_new, data_tt_new])

# Merge datanew with data_bas to get baseline std_energy
datanew['std_energy(kWh)'] = datanew['std_energy(kWh)'].astype(float)
data_bas['std_energy(kWh)'] = data_bas['std_energy(kWh)'].astype(float)

datanew = datanew.merge(data_bas[['std_energy(kWh)','In_ch']], on='In_ch', suffixes=('', '_bas'))

# Calculate combined standard deviation
datanew['combined_std_energy'] = np.sqrt(datanew['std_energy(kWh)']**2 + datanew['std_energy(kWh)_bas']**2)


# #%% Make visualisation of calculated mac's
# datacp=total_df_inch[total_df_inch['Dec']=='cp']
# datatt=total_df_inch[total_df_inch['Dec']=='tt']
# datatucker=total_df_inch[total_df_inch['Dec']=='tucker']


# # inputs
# compression_ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
# input_channels = [192, 256, 320, 384]

# #per compression
# compr1_cp=datacp[datacp['Comp']==0.1]
# compr25_cp=datacp[datacp['Comp']==0.25]
# compr5_cp=datacp[datacp['Comp']==0.5]
# compr75_cp=datacp[datacp['Comp']==0.75]
# compr9_cp=datacp[datacp['Comp']==0.9]

# # MAC operations data for each method at each compression ratio
# cp_macs = {
#     0.1: (compr1_cp['MAC_original']-compr1_cp['MAC']),
#     0.25: (compr25_cp['MAC_original']-compr25_cp['MAC']),
#     0.5: (compr5_cp['MAC_original']-compr5_cp['MAC']),
#     0.75: (compr75_cp['MAC_original']-compr75_cp['MAC']),
#     0.9: (compr9_cp['MAC_original']-compr9_cp['MAC'])
# }

# #per compression
# compr1_tucker=datatucker[datatucker['Comp']==0.1]
# compr25_tucker=datatucker[datatucker['Comp']==0.25]
# compr5_tucker=datatucker[datatucker['Comp']==0.5]
# compr75_tucker=datatucker[datatucker['Comp']==0.75]
# compr9_tucker=datatucker[datatucker['Comp']==0.9]

# # MAC operations data for each method at each compression ratio
# tucker_macs = {
#     0.1: (compr1_tucker['MAC_original']-compr1_tucker['MAC']),
#     0.25: (compr25_tucker['MAC_original']-compr25_tucker['MAC']),
#     0.5: (compr5_tucker['MAC_original']-compr5_tucker['MAC']),
#     0.75: (compr75_tucker['MAC_original']-compr75_tucker['MAC']),
#     0.9: (compr9_tucker['MAC_original']-compr9_tucker['MAC'])
# }
# #per compression
# compr1_tt=datatt[datatt['Comp']==0.1]
# compr25_tt=datatt[datatt['Comp']==0.25]
# compr5_tt=datatt[datatt['Comp']==0.5]
# compr75_tt=datatt[datatt['Comp']==0.75]
# compr9_tt=datatt[datatt['Comp']==0.9]

# # MAC operations data for each method at each compression ratio
# tt_macs = {
#     0.1: (compr1_tt['MAC_original']-compr1_tt['MAC']),
#     0.25: (compr25_tt['MAC_original']-compr25_tt['MAC']),
#     0.5: (compr5_tt['MAC_original']-compr5_tt['MAC']),
#     0.75: (compr75_tt['MAC_original']-compr75_tt['MAC']),
#     0.9: (compr9_tt['MAC_original']-compr9_tt['MAC'])
# }
# # Set up the figure with subplots for each method
# fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

# bar_width = 0.1
# x = np.arange(len(input_channels))

# # Color definitions
# colors = {
#     0.1: '#4477aa',  # blue
#     0.25: '#66ccee', # cyan
#     0.5: '#228833',  # green
#     0.75: '#d4bb44', # yellow
#     0.9: '#ee6677'   # red
# }

# # Define a function to plot data for each method
# def plot_method(ax, method_macs, method_name):
#     for i, cr in enumerate(compression_ratios):
#         offset = (i - 2) * (bar_width + 0.01)  # Adjust the offset for each bar within a group
#         bars = ax.bar(x + offset, method_macs[cr], width=bar_width, color=colors[cr],label=f'{cr}',zorder=2)
#         # for bar in bars:
#         #     ax.annotate(f'{cr}',
#         #                 xy=(bar.get_x() + bar.get_width() / 2, 0),
#         #                 xytext=(0, -12),  # 12 points vertical offset
#         #                 textcoords="offset points",
#         #                 ha='center', va='top', fontsize=10, color='black')  # Increased fontsize
    
#     ax.grid( zorder=1)    
#     #ax.set_ylabel('MAC reduction ratio')
#     ax.set_ylim([mac_min-0.1,mac_max+0.1])
#     ax.set_title(f'{method_name}')
#     ax.set_xticks(x)
#     ax.set_xticklabels(input_channels)
#     ax.tick_params(axis='x', pad=10)  # Move the x-tick labels lower
#     ax.grid(True, which='both', zorder=1)
# # Plot data for each method
# plot_method(axes[0], cp_macs, 'MAC reduction for diff. S for CP decomposition')
# plot_method(axes[1], tucker_macs, 'MAC reduction for diff. S for Tucker decomposition')
# plot_method(axes[2], tt_macs, 'MAC reduction for diff. S for TT decomposition')
# plt.subplots_adjust(left=0.25)

# fig.text(0.02, 0.5, 'Reduced number of MAC operations', va='center', rotation='vertical', fontsize=12)

# # Add common x-label
# plt.xlabel('Number of Input Channels', fontsize=12)
# plt.legend(loc ="lower right", ncol=5)
# plt.tight_layout()
# plt.show()


# #%% Make visualisation of meas mem's
# mac_min=np.min((total_df_inch['Mem_meas'].to_numpy()))
# mac_max=np.max((total_df_inch['Mem_meas'].to_numpy()))


# datacp=total_df_inch[total_df_inch['Dec']=='cp']
# datatt=total_df_inch[total_df_inch['Dec']=='tt']
# datatucker=total_df_inch[total_df_inch['Dec']=='tucker']


# # inputs
# compression_ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
# input_channels = [192, 256, 320, 384]

# #per compression
# compr1_cp=datacp[datacp['Comp']==0.1]
# compr25_cp=datacp[datacp['Comp']==0.25]
# compr5_cp=datacp[datacp['Comp']==0.5]
# compr75_cp=datacp[datacp['Comp']==0.75]
# compr9_cp=datacp[datacp['Comp']==0.9]

# # MAC operations data for each method at each compression ratio
# cp_macs = {
#     0.1: compr1_cp['Mem_meas'],
#     0.25: compr25_cp['Mem_meas'],
#     0.5: compr5_cp['Mem_meas'],
#     0.75: compr75_cp['Mem_meas'],
#     0.9: compr9_cp['Mem_meas']
# }

# #per compression
# compr1_tucker=datatucker[datatucker['Comp']==0.1]
# compr25_tucker=datatucker[datatucker['Comp']==0.25]
# compr5_tucker=datatucker[datatucker['Comp']==0.5]
# compr75_tucker=datatucker[datatucker['Comp']==0.75]
# compr9_tucker=datatucker[datatucker['Comp']==0.9]

# # MAC operations data for each method at each compression ratio
# tucker_macs = {
#     0.1: compr1_tucker['Mem_meas'],
#     0.25: compr25_tucker['Mem_meas'],
#     0.5: compr5_tucker['Mem_meas'],
#     0.75: compr75_tucker['Mem_meas'],
#     0.9: compr9_tucker['Mem_meas']
# }
# #per compression
# compr1_tt=datatt[datatt['Comp']==0.1]
# compr25_tt=datatt[datatt['Comp']==0.25]
# compr5_tt=datatt[datatt['Comp']==0.5]
# compr75_tt=datatt[datatt['Comp']==0.75]
# compr9_tt=datatt[datatt['Comp']==0.9]

# # MAC operations data for each method at each compression ratio
# tt_macs = {
#     0.1: compr1_tt['Mem_meas'],
#     0.25: compr25_tt['Mem_meas'],
#     0.5: compr5_tt['Mem_meas'],
#     0.75: compr75_tt['Mem_meas'],
#     0.9: compr9_tt['Mem_meas']
# }
# # Set up the figure with subplots for each method
# fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

# bar_width = 0.1
# x = np.arange(len(input_channels))

# # Color definitions
# colors = {
#     0.1: '#4477aa',  # blue
#     0.25: '#66ccee', # cyan
#     0.5: '#228833',  # green
#     0.75: '#d4bb44', # yellow
#     0.9: '#ee6677'   # red
# }

# # Define a function to plot data for each method
# def plot_method(ax, method_macs, method_name):
#     for i, cr in enumerate(compression_ratios):
#         offset = (i - 2) * (bar_width + 0.01)  # Adjust the offset for each bar within a group
#         bars = ax.bar(x + offset, method_macs[cr], width=bar_width, color=colors[cr],label=f'{cr}',zorder=2)
#         # for bar in bars:
#         #     ax.annotate(f'{cr}',
#         #                 xy=(bar.get_x() + bar.get_width() / 2, 0),
#         #                 xytext=(0, -12),  # 12 points vertical offset
#         #                 textcoords="offset points",
#         #                 ha='center', va='top', fontsize=10, color='black')  # Increased fontsize
    
#     ax.grid( zorder=1)    
#     #ax.set_ylabel('Memory (# params)')
#     ax.set_ylim([mac_min-0.1,mac_max+0.1])
#     ax.set_title(f'{method_name}')
#     ax.set_xticks(x)
#     ax.set_xticklabels(input_channels)
#     ax.tick_params(axis='x', pad=10)  # Move the x-tick labels lower
#     ax.grid(True, which='both', zorder=1)
# # Plot data for each method
# plot_method(axes[0], cp_macs, 'Measured memory for diff. S for CP decomposition')
# plot_method(axes[1], tucker_macs, 'Measured memory for diff. S for Tucker decomposition')
# plot_method(axes[2], tt_macs, 'Measured memory for diff. S for TT decomposition')

# plt.subplots_adjust(left=0.25)

# fig.text(0.02, 0.5, 'Memory (MB)', va='center', rotation='vertical', fontsize=12)

# # Add common x-label
# plt.xlabel('Number of Input Channels', fontsize=12)
# plt.legend(loc ="best", ncol=5)
# plt.tight_layout()
# plt.show()
# #%% Memory plots
# mac_min=np.min((total_df_inch['Memcalc_diff'].to_numpy()))-0.2*10**6
# mac_max=np.max((total_df_inch['Memcalc_diff'].to_numpy()))+0.2*10**6


# datacp=total_df_inch[total_df_inch['Dec']=='cp']
# datatt=total_df_inch[total_df_inch['Dec']=='tt']
# datatucker=total_df_inch[total_df_inch['Dec']=='tucker']


# # inputs
# compression_ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
# input_channels = [192, 256, 320, 384]

# #per compression
# compr1_cp=datacp[datacp['Comp']==0.1]
# compr25_cp=datacp[datacp['Comp']==0.25]
# compr5_cp=datacp[datacp['Comp']==0.5]
# compr75_cp=datacp[datacp['Comp']==0.75]
# compr9_cp=datacp[datacp['Comp']==0.9]

# # MAC operations data for each method at each compression ratio
# cp_macs = {
#     0.1: compr1_cp['Memcalc_diff'],
#     0.25: compr25_cp['Memcalc_diff'],
#     0.5: compr5_cp['Memcalc_diff'],
#     0.75: compr75_cp['Memcalc_diff'],
#     0.9: compr9_cp['Memcalc_diff']
# }

# #per compression
# compr1_tucker=datatucker[datatucker['Comp']==0.1]
# compr25_tucker=datatucker[datatucker['Comp']==0.25]
# compr5_tucker=datatucker[datatucker['Comp']==0.5]
# compr75_tucker=datatucker[datatucker['Comp']==0.75]
# compr9_tucker=datatucker[datatucker['Comp']==0.9]

# # MAC operations data for each method at each compression ratio
# tucker_macs = {
#     0.1: compr1_tucker['Memcalc_diff'],
#     0.25: compr25_tucker['Memcalc_diff'],
#     0.5: compr5_tucker['Memcalc_diff'],
#     0.75: compr75_tucker['Memcalc_diff'],
#     0.9: compr9_tucker['Memcalc_diff']
# }
# #per compression
# compr1_tt=datatt[datatt['Comp']==0.1]
# compr25_tt=datatt[datatt['Comp']==0.25]
# compr5_tt=datatt[datatt['Comp']==0.5]
# compr75_tt=datatt[datatt['Comp']==0.75]
# compr9_tt=datatt[datatt['Comp']==0.9]

# # MAC operations data for each method at each compression ratio
# tt_macs = {
#     0.1: compr1_tt['Memcalc_diff'],
#     0.25: compr25_tt['Memcalc_diff'],
#     0.5: compr5_tt['Memcalc_diff'],
#     0.75: compr75_tt['Memcalc_diff'],
#     0.9: compr9_tt['Memcalc_diff']
# }
# # Set up the figure with subplots for each method
# fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

# bar_width = 0.1
# x = np.arange(len(input_channels))

# # Color definitions
# colors = {
#     0.1: '#4477aa',  # blue
#     0.25: '#66ccee', # cyan
#     0.5: '#228833',  # green
#     0.75: '#d4bb44', # yellow
#     0.9: '#ee6677'   # red
# }

# # Define a function to plot data for each method
# def plot_method(ax, method_macs, method_name):
#     for i, cr in enumerate(compression_ratios):
#         offset = (i - 2) * (bar_width + 0.01)  # Adjust the offset for each bar within a group
#         bars = ax.bar(x + offset, method_macs[cr], width=bar_width, color=colors[cr],label=f'{cr}',zorder=2)
#         # for bar in bars:
#         #     ax.annotate(f'{cr}',
#         #                 xy=(bar.get_x() + bar.get_width() / 2, 0),
#         #                 xytext=(0, -12),  # 12 points vertical offset
#         #                 textcoords="offset points",
#         #                 ha='center', va='top', fontsize=10, color='black')  # Increased fontsize
    
#     ax.grid( zorder=1)    
#     #ax.set_ylabel('Memory (# params)')
#     ax.set_ylim([mac_min-0.1,mac_max+0.1])
#     ax.set_title(f'{method_name}')
#     ax.set_xticks(x)
#     ax.set_xticklabels(input_channels)
#     ax.tick_params(axis='x', pad=10)  # Move the x-tick labels lower
#     ax.grid(True, which='both', zorder=1)
# # Plot data for each method
# plot_method(axes[0], cp_macs, 'Memory for diff. S for CP decomposition')
# plot_method(axes[1], tucker_macs, 'Memory for diff. S for Tucker decomposition')
# plot_method(axes[2], tt_macs, 'Memory for diff. S for TT decomposition')

# plt.subplots_adjust(left=0.25)

# fig.text(0.02, 0.5, 'Memory (#params)', va='center', rotation='vertical', fontsize=12)

# # Add common x-label
# plt.xlabel('Number of Input Channels', fontsize=12)
# plt.legend(loc ="best", ncol=5)
# plt.tight_layout()
# plt.show()

# #%% Plot measured memory and energy consumption

# # Load the data
# # datanew = pd.read_csv('/mnt/data/your_data.csv')
# Define base colors for each input channel


input_channels = [192, 256, 320, 384]
colors = ['#4477aa', '#66ccee', '#228833', '#ee6677']


# Set up the figure and subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
methods = datanew['Dec'].unique()

# Plot data for each decomposition method in a separate subplot
for ax, method in zip(axes, methods):
    method_data = datanew[datanew['Dec'] == method]
    for color, in_ch in zip(colors, input_channels):
        subset = method_data[method_data['In_ch'] == in_ch]
        # Plot the line
        sns.lineplot(
            data=subset,
            x='Comp',
            y='delta_energy(kWh)',
            ax=ax,
            label=f'In_ch={in_ch}',
            color=color,
            marker='o'
        )
        # Shade the area representing the standard deviation
        ax.fill_between(
            subset['Comp'],
            subset['delta_energy(kWh)'] - subset['combined_std_energy'],
            subset['delta_energy(kWh)'] + subset['combined_std_energy'],
            color=color,
            alpha=0.3
        )
        if method == 'cp':
            title = 'CP'
            ax.legend().remove()

        elif method == 'tucker':
            title = 'Tucker'
            ax.legend().remove()
        elif method == 'tt':
            title = 'TT'
            plt.legend(title='# input channels')
    ax.set_title(f'{title} Decomposition')
    ax.set_xlabel('Compression Ratio')
axes[0].set_ylabel('Energy saved (kWh)')


# Add a main title
plt.suptitle('Energy Consumption Across Compression Ratios by Input Channels and Decomposition Methods', y=0.98)

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 0.99, 1])

# Save the plot as an SVG file
plt.savefig('C:/Users/demib/Documents/Thesis/Data_final_processing/Figures/Desktop/dp_eng_inch.pdf', format='pdf')

# Show the plot
plt.show()



#%%

# Calculate MAC reductions and memory for each method
compression_ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
input_channels = [192, 256, 320, 384]

datacp = total_df_inch[total_df_inch['Dec'] == 'cp']
datatt = total_df_inch[total_df_inch['Dec'] == 'tt']
datatucker = total_df_inch[total_df_inch['Dec'] == 'tucker']

cp_macs = calculate_mac_reduction(datacp, compression_ratios)
tucker_macs = calculate_mac_reduction(datatucker, compression_ratios)
tt_macs = calculate_mac_reduction(datatt, compression_ratios)

cp_memory_meas = calculate_memory(datacp, compression_ratios, 'Mem_meas_diff')
tucker_memory_meas = calculate_memory(datatucker, compression_ratios, 'Mem_meas_diff')
tt_memory_meas = calculate_memory(datatt, compression_ratios, 'Mem_meas_diff')

cp_memory_calc_diff = calculate_memory(datacp, compression_ratios, 'Memcalc_diff')
tucker_memory_calc_diff = calculate_memory(datatucker, compression_ratios, 'Memcalc_diff')
tt_memory_calc_diff = calculate_memory(datatt, compression_ratios, 'Memcalc_diff')



# Color definitions
colors = {
    0.1: '#4477aa',  # blue
    0.25: '#66ccee', # cyan
    0.5: '#228833',  # green
    0.75: '#d4bb44', # yellow
    0.9: '#ee6677'   # red
}

# Set up the figure with subplots for each method
fig, axes = plt.subplots(1,3, figsize=(13,4), sharex=True, gridspec_kw={'wspace': 0.3})

# Define y-limits for each column
ylim_macs = [-1e9, 1.0e9]
ylim_memory = [0, 160]
ylim_memory_diff = [0, 6.45e6]

# # Plot MAC reduction data for each method
# plot_method(axes[0, 0], cp_macs, 'MAC reduction CP', colors, compression_ratios, input_channels, ylim=ylim_macs)
# plot_method(axes[1, 0], tucker_macs, 'MAC reduction Tucker', colors, compression_ratios, input_channels, ylim=ylim_macs)
# plot_method(axes[2, 0], tt_macs, 'MAC reduction TT', colors, compression_ratios, input_channels, ylim=ylim_macs)

# # Plot measured memory data for each method
# plot_method(axes[0, 1], cp_memory_meas, 'Measured memory CP', colors, compression_ratios, input_channels, ylim=ylim_memory)
# plot_method(axes[1, 1], tucker_memory_meas, 'Measured memory Tucker', colors, compression_ratios, input_channels, ylim=ylim_memory)
# plot_method(axes[2, 1], tt_memory_meas, 'Measured memory TT', colors, compression_ratios, input_channels, ylim=ylim_memory)

# # Plot calculated memory difference data for each method
# plot_method(axes[0, 2], cp_memory_calc_diff, 'Calculated memory CP', colors, compression_ratios, input_channels, ylim=ylim_memory_diff)
# plot_method(axes[1, 2], tucker_memory_calc_diff, 'Calculated memory Tucker', colors, compression_ratios, input_channels, ylim=ylim_memory_diff)
# plot_method(axes[2, 2], tt_memory_calc_diff, 'Calculated memory TT', colors, compression_ratios, input_channels, ylim=ylim_memory_diff)

# # Add common y-labels for each column
# fig.text(0.015, 0.5, 'Reduced number of MAC operations', va='center', rotation='vertical', fontsize=12)
# fig.text(0.345, 0.5, 'Memory (MB)', va='center', rotation='vertical', fontsize=12)
# fig.text(0.685, 0.5, 'Memory (#params)', va='center', rotation='vertical', fontsize=12)

# # Add common x-label
# fig.text(0.5, 0.04, 'Number of Input Channels', ha='center', fontsize=12)

# handles, labels = axes[0, 0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right', ncol=5, title='Compression Ratio')
# plt.suptitle('Experiment 1: Memory and Computation Complexity', x=0.02, y=0.96, ha='left', fontsize=14, fontweight='bold')

# plt.subplots_adjust(left=0.07, right=0.98, top=0.87, bottom=0.1, wspace=0.3, hspace=0.3)
# Plot measured memory data for each method
plot_method(axes[0], cp_memory_meas, 'CP', colors, compression_ratios, input_channels, ylim=ylim_memory)
plot_method(axes[1], tucker_memory_meas, 'Tucker', colors, compression_ratios, input_channels, ylim=ylim_memory)
plot_method(axes[2], tt_memory_meas, 'TT', colors, compression_ratios, input_channels, ylim=ylim_memory)

# Add common y-label for the plots
fig.text(0.02, 0.5, 'Memory (MB)', va='center', rotation='vertical', fontsize=12)
plt.subplots_adjust(left=0.06, right=0.98, top=0.87, bottom=0.1, wspace=0.3, hspace=0.3)
plt.legend(title='Compression ratio')
# Add common x-label
fig.text(0.5, 0.005, 'Input channels', ha='center', fontsize=12)

# Add legend
handles, labels = axes[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='center right', ncol=2, title='Compression Ratio')
# Add a common title
#plt.suptitle('Experiment 3: Memory and Computation Complexity', x=0.5, y=0.98, ha='center', fontsize=14, fontweight='bold')
plt.suptitle('Measured Memory for Experiment 1 on GPU')


# Save the plot as an SVG file
svg_filename = 'C:/Users/demib/Documents/Thesis/Data_final_processing/Figures/Desktop/dp_mem_inch.pdf'
plt.savefig(svg_filename, format='pdf')


plt.show()


save_path = "inchall_reg.pkl"
with open(save_path, 'wb') as f:
    pickle.dump(datanew, f)
