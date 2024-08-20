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
    out = int(parts[0][5:])
    
    # Assign a numeric value to compression types to ensure correct order
    compression_order = {'factcp': 0, 'facttucker': 1, 'facttt': 2}
    
    # Return a tuple that will be used for sorting
    return (compression_order[compression_type],r_value, out )

def process_data(total_df_outch, data_bas, dec_method):
    # Filter data for the specific decomposition method
    data_tt = total_df_outch[total_df_outch['Dec'] == dec_method]
    
    # Initialize an empty list to store processed data
    processed_data = []

    # Define the input channels to process
    out_channels = [192, 256, 320, 384]
    
    for out_ch in out_channels:
        # Filter data for the specific input channel
        data_outch = data_tt[data_tt['Out_ch'] == out_ch]
        data_bas_outch = data_bas[data_bas['Out_ch'] == out_ch]
        
        # Convert energy(kWh) to numeric
        data_outch.loc[:, 'energy(kWh)'] = pd.to_numeric(data_outch['energy(kWh)'], errors='coerce')
        
        # Calculate delta energy
        data_outch['delta_energy(kWh)'] = np.ones(len(data_outch['energy(kWh)'])) * data_bas_outch['energy(kWh)'].to_numpy() - data_outch['energy(kWh)']
        data_outch.loc[:, 'delta_energy(kWh)'] = pd.to_numeric(data_outch['delta_energy(kWh)'], errors='coerce')
        
        # Calculate percentage energy
        data_outch['per_energy(kWh)'] = data_outch['delta_energy(kWh)'] / (np.ones(len(data_outch['energy(kWh)'])) * data_bas_outch['energy(kWh)'].to_numpy()) * 100
        data_outch.loc[:, 'per_energy(kWh)'] = pd.to_numeric(data_outch.loc[:, 'per_energy(kWh)'], errors='coerce')
        data_outch.loc[:, 'per_energy(kWh)'] = np.around(data_outch.loc[:, 'per_energy(kWh)'], decimals=2)
        
        # Append the processed data to the list
        processed_data.append(data_outch)
    
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
allinfo_outch=pd.read_pickle('allinfo_out.pkl')

data_tot=pd.DataFrame.from_dict(pd.read_pickle('data_tt_dec_outch.pkl')).transpose()

sort_keys=sorted(data_tot.index, key=custom_sort_key)
data_tot = data_tot.loc[sort_keys]


data_bas=pd.DataFrame.from_dict(pd.read_pickle('data_bas_outch.pkl')).transpose()
data_bas['Out_ch'] = data_bas.index.to_series().str.extract(r'outch(\d+)')[0].astype(float)

# Create the new index based on the specified format
new_index = [
    f"outch{row['Out_ch']}-inch{row['In_ch']}-fact{row['Dec']}-r{row['Comp']}-wh{row['In_feat'][2]}"
    for idx, row in allinfo_outch.iterrows()
]

# Assign the new index to the DataFrame
allinfo_outch.index = new_index

#load memory measurements
mem_outch=pd.read_pickle('mem_outch (1).pkl')
mem_outch=mem_outch.rename(columns={'Mem':'Mem_meas'})
mem_outch=mem_outch.loc[sort_keys]

mem_outch.index=new_index

mem_bas=pd.read_pickle('mem_bas_outch.pkl')
membas_tot=pd.concat([mem_bas] * 15, ignore_index=True)


total_df_outch=pd.concat([data_tot,allinfo_outch,mem_outch], axis=1)
total_df_outch['Mem_meas_diff']=total_df_outch['Mem_meas']-membas_tot['Mem'].to_numpy()


#%%

data_tucker_new = process_data(total_df_outch, data_bas, 'tucker')
data_cp_new = process_data(total_df_outch, data_bas, 'cp')
data_tt_new = process_data(total_df_outch, data_bas, 'tt')

datanew=pd.concat([data_cp_new, data_tucker_new, data_tt_new])

# Merge datanew with data_bas to get baseline std_energy
datanew['std_energy(kWh)'] = datanew['std_energy(kWh)'].astype(float)
data_bas['std_energy(kWh)'] = data_bas['std_energy(kWh)'].astype(float)

datanew = datanew.merge(data_bas[['std_energy(kWh)','Out_ch']], on='Out_ch', suffixes=('', '_bas'))

# Calculate combined standard deviation
datanew['combined_std_energy'] = np.sqrt(datanew['std_energy(kWh)']**2 + datanew['std_energy(kWh)_bas']**2)



output_channels = [192, 256, 320, 384]
colors = ['#4477aa', '#66ccee', '#228833', '#ee6677']


# Set up the figure and subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
methods = datanew['Dec'].unique()

# Plot data for each decomposition method in a separate subplot
for ax, method in zip(axes, methods):
    method_data = datanew[datanew['Dec'] == method]
    for color, out_ch in zip(colors, output_channels):
        subset = method_data[method_data['Out_ch'] == out_ch]
        # Plot the line
        sns.lineplot(
            data=subset,
            x='Comp',
            y='delta_energy(kWh)',
            ax=ax,
            label=f'Out_ch={out_ch}',
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
            ax.legend(title='# output channels')
            title = 'TT'
    ax.set_title(f'{title} Decomposition')
    ax.set_xlabel('Compression Ratio')
axes[0].set_ylabel('Energy saved (kWh)')


# Add a main title
plt.suptitle('Energy Consumption Across Compression Ratios by Output Channels and Decomposition Methods', y=0.98)

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 0.99, 1])

# Save the plot as an SVG file
plt.savefig('C:/Users/demib/Documents/Thesis/Data_final_processing/Figures/Desktop/dp_eng_outch.pdf', format='pdf')

# Show the plot
plt.show()

#%%
# Calculate MAC reductions and memory for each method
compression_ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
out_channels = [192, 256, 320, 384]

datacp = total_df_outch[total_df_outch['Dec'] == 'cp']
datatt = total_df_outch[total_df_outch['Dec'] == 'tt']
datatucker = total_df_outch[total_df_outch['Dec'] == 'tucker']

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
fig, axes = plt.subplots(1, 3, figsize=(13,4), sharex=True)

# Define y-limits for each column
ylim_macs = [-0.3e9, 0.35e9]
ylim_memory = [0, 65]
ylim_memory_diff = [0, 3.9e6]

# # Plot MAC reduction data for each method
# plot_method(axes[0, 0], cp_macs, 'MAC reduction CP', colors, compression_ratios, out_channels, ylim=ylim_macs)
# plot_method(axes[1, 0], tucker_macs, 'MAC reduction Tucker', colors, compression_ratios, out_channels, ylim=ylim_macs)
# plot_method(axes[2, 0], tt_macs, 'MAC reduction TT', colors, compression_ratios, out_channels, ylim=ylim_macs)

# # Plot measured memory data for each method
# plot_method(axes[0, 1], cp_memory_meas, 'Measured memory CP', colors, compression_ratios, out_channels, ylim=ylim_memory)
# plot_method(axes[1, 1], tucker_memory_meas, 'Measured memory Tucker', colors, compression_ratios, out_channels, ylim=ylim_memory)
# plot_method(axes[2, 1], tt_memory_meas, 'Measured memory TT', colors, compression_ratios, out_channels, ylim=ylim_memory)

# # Plot calculated memory difference data for each method
# plot_method(axes[0, 2], cp_memory_calc_diff, 'Calculated memory CP', colors, compression_ratios, out_channels, ylim=ylim_memory_diff)
# plot_method(axes[1, 2], tucker_memory_calc_diff, 'Calculated memory Tucker', colors, compression_ratios, out_channels, ylim=ylim_memory_diff)
# plot_method(axes[2, 2], tt_memory_calc_diff, 'Calculated memory TT', colors, compression_ratios, out_channels, ylim=ylim_memory_diff)

# # Add common y-labels for each column
# fig.text(0.015, 0.5, 'Reduced number of MAC operations', va='center', rotation='vertical', fontsize=12)
# fig.text(0.345, 0.5, 'Memory (MB)', va='center', rotation='vertical', fontsize=12)
# fig.text(0.685, 0.5, 'Memory (#params)', va='center', rotation='vertical', fontsize=12)

# # Add common x-label
# fig.text(0.5, 0.04, 'Number of Output Channels', ha='center', fontsize=12)

# handles, labels = axes[0, 0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right', ncol=5, title='Compression Ratio')
# plt.suptitle('Experiment 2: Memory and Computation Complexity', x=0.02, y=0.96, ha='left', fontsize=14, fontweight='bold')

# plt.subplots_adjust(left=0.07, right=0.98, top=0.87, bottom=0.1, wspace=0.3, hspace=0.3)

plot_method(axes[0], cp_memory_meas, 'CP', colors, compression_ratios, out_channels, ylim=ylim_memory)
plot_method(axes[1], tucker_memory_meas, 'Tucker', colors, compression_ratios, out_channels, ylim=ylim_memory)
plot_method(axes[2], tt_memory_meas, 'TT', colors, compression_ratios, out_channels, ylim=ylim_memory)

# Add common y-label for the plots
fig.text(0.02, 0.5, 'Memory (MB)', va='center', rotation='vertical', fontsize=12)
plt.subplots_adjust(left=0.06, right=0.98, top=0.87, bottom=0.1, wspace=0.3, hspace=0.3)

# Add common x-label
fig.text(0.5, 0.005, 'Out channels', ha='center', fontsize=12)

# Add legend
handles, labels = axes[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='center right', ncol=2, title='Compression Ratio')
# Add a common title
#plt.suptitle('Experiment 3: Memory and Computation Complexity', x=0.5, y=0.98, ha='center', fontsize=14, fontweight='bold')
plt.suptitle('Measured Memory for Experiment 2 on GPU')
plt.legend(title='Compression ratio')

# Save the plot as an SVG file
svg_filename = 'C:/Users/demib/Documents/Thesis/Data_final_processing/Figures/Desktop/dp_mem_outch.pdf'
plt.savefig(svg_filename, format='pdf')


plt.show()

# datacp=total_df_outch[total_df_outch['Dec']=='cp']
# datatt=total_df_outch[total_df_outch['Dec']=='tt']
# datatucker=total_df_outch[total_df_outch['Dec']=='tucker']

# mac_min=np.min((total_df_outch['MAC_original']-total_df_outch['MAC']))-0.2*10**8
# mac_max=np.max((total_df_outch['MAC_original']-total_df_outch['MAC']))+0.2*10**8

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

# plot_method(axes[0], cp_macs, 'MAC reduction for diff. T for CP decomposition')
# plot_method(axes[1], tucker_macs, 'MAC reduction for diff. T for Tucker decomposition')
# plot_method(axes[2], tt_macs, 'MAC reduction for diff. T for TT decomposition')

# # Adjust subplot parameters to create space on the left
# plt.subplots_adjust(left=0.25)

# fig.text(0.02, 0.5, 'Reduced number of MAC operations', va='center', rotation='vertical', fontsize=12)


# # Add common x-label
# plt.xlabel('Number of Output Channels', fontsize=12)
# plt.legend(loc ="lower right", ncol=5)
# plt.tight_layout()
# plt.show()


# #%% Memory plots
# mac_min=np.min((total_df_outch['Memcalc_diff'].to_numpy()))-0.2*10**6
# mac_max=np.max((total_df_outch['Memcalc_diff'].to_numpy()))+0.2*10**6


# datacp=total_df_outch[total_df_outch['Dec']=='cp']
# datatt=total_df_outch[total_df_outch['Dec']=='tt']
# datatucker=total_df_outch[total_df_outch['Dec']=='tucker']


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
# plot_method(axes[0], cp_macs, 'Memory for diff. T for CP decomposition')
# plot_method(axes[1], tucker_macs, 'Memory for diff. T for Tucker decomposition')
# plot_method(axes[2], tt_macs, 'Memory for diff. T for TT decomposition')

# plt.subplots_adjust(left=0.25)

# fig.text(0.02, 0.5, 'Memory (# params)', va='center', rotation='vertical', fontsize=12)


# # Add common x-label
# plt.xlabel('Number of Output Channels', fontsize=12)
# plt.legend(loc ="best", ncol=5)
# plt.tight_layout()
# plt.show()


# #%%Let's create plots to show difference in Out_channels

# data_tt=total_df_outch[total_df_outch['Dec']=='tucker']

# data_outch192=data_tt[data_tt['Out_ch']==192]
# data_bas192=data_bas[data_bas['Out_ch']==192]
# data_outch256=data_tt[data_tt['Out_ch']==256]
# data_bas256=data_bas[data_bas['Out_ch']==256]
# data_outch320=data_tt[data_tt['Out_ch']==320]
# data_bas320=data_bas[data_bas['Out_ch']==320]
# data_outch384=data_tt[data_tt['Out_ch']==384]
# data_bas384=data_bas[data_bas['Out_ch']==384]

# data_outch192.loc[:, 'energy(kWh)'] = pd.to_numeric(data_outch192['energy(kWh)'], errors='coerce')
# data_outch192['delta_energy(kWh)']=np.ones(len(data_outch192['energy(kWh)']))*data_bas192['energy(kWh)'].to_numpy()-data_outch192['energy(kWh)']
# data_outch192.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_outch192['delta_energy(kWh)'], errors='coerce')
# data_outch192['per_energy(kWh)']=data_outch192['delta_energy(kWh)']/(np.ones(len(data_outch192['energy(kWh)']))*data_bas192['energy(kWh)'].to_numpy())*100
# data_outch192.loc[:,'per_energy(kWh)']=pd.to_numeric(data_outch192.loc[:,'per_energy(kWh)'],errors='coerce')
# data_outch192.loc[:,'per_energy(kWh)']=np.around(data_outch192.loc[:,'per_energy(kWh)'], decimals=2)

# data_outch256.loc[:, 'energy(kWh)'] = pd.to_numeric(data_outch256['energy(kWh)'], errors='coerce')
# data_outch256['delta_energy(kWh)']=np.ones(len(data_outch256['energy(kWh)']))*data_bas256['energy(kWh)'].to_numpy()-data_outch256['energy(kWh)']
# data_outch256.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_outch256['delta_energy(kWh)'], errors='coerce')
# data_outch256['per_energy(kWh)']=data_outch256['delta_energy(kWh)']/(np.ones(len(data_outch256['energy(kWh)']))*data_bas256['energy(kWh)'].to_numpy())*100
# data_outch256.loc[:,'per_energy(kWh)']=pd.to_numeric(data_outch256.loc[:,'per_energy(kWh)'],errors='coerce')
# data_outch256.loc[:,'per_energy(kWh)']=np.around(data_outch256.loc[:,'per_energy(kWh)'], decimals=2)

# data_outch320.loc[:, 'energy(kWh)'] = pd.to_numeric(data_outch320['energy(kWh)'], errors='coerce')
# data_outch320['delta_energy(kWh)']=np.ones(len(data_outch320['energy(kWh)']))*data_bas320['energy(kWh)'].to_numpy()-data_outch320['energy(kWh)']
# data_outch320.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_outch320['delta_energy(kWh)'], errors='coerce')
# data_outch320['per_energy(kWh)']=data_outch320['delta_energy(kWh)']/(np.ones(len(data_outch320['energy(kWh)']))*data_bas320['energy(kWh)'].to_numpy())*100
# data_outch320.loc[:,'per_energy(kWh)']=pd.to_numeric(data_outch320.loc[:,'per_energy(kWh)'],errors='coerce')
# data_outch320.loc[:,'per_energy(kWh)']=np.around(data_outch320.loc[:,'per_energy(kWh)'], decimals=2)

# data_outch384.loc[:, 'energy(kWh)'] = pd.to_numeric(data_outch384['energy(kWh)'], errors='coerce')
# data_outch384['delta_energy(kWh)']=np.ones(len(data_outch384['energy(kWh)']))*data_bas384['energy(kWh)'].to_numpy()-data_outch384['energy(kWh)']
# data_outch384.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_outch384['delta_energy(kWh)'], errors='coerce')
# data_outch384['per_energy(kWh)']=data_outch384['delta_energy(kWh)']/(np.ones(len(data_outch384['energy(kWh)']))*data_bas384['energy(kWh)'].to_numpy())*100
# data_outch384.loc[:,'per_energy(kWh)']=pd.to_numeric(data_outch384.loc[:,'per_energy(kWh)'],errors='coerce')
# data_outch384.loc[:,'per_energy(kWh)']=np.around(data_outch384.loc[:,'per_energy(kWh)'], decimals=2)


# z192 = np.polyfit(data_outch192['Comp'], (data_outch192['MAC_original']-data_outch192['MAC'])/data_outch192['MAC_original']*100, 2)
# p192 = np.poly1d(z192)
# xp192 = np.linspace(0.1, 0.9, 100)

# z256 = np.polyfit(data_outch256['Comp'], (data_outch256['MAC_original']-data_outch256['MAC'])/data_outch256['MAC_original']*100, 2)
# p256 = np.poly1d(z256)
# xp256 = np.linspace(0.1, 0.9, 100)


# z320 = np.polyfit(data_outch320['Comp'],( data_outch320['MAC_original']-data_outch320['MAC'])/data_outch320['MAC_original']*100, 2)
# p320 = np.poly1d(z320)
# xp320 = np.linspace(0.1, 0.9, 100)

# z384 = np.polyfit(data_outch384['Comp'],( data_outch384['MAC_original']-data_outch384['MAC'])/data_outch384['MAC_original']*100, 2)
# p384 = np.poly1d(z384)
# xp384 = np.linspace(0.1, 0.9, 100)


# plt.figure()
# plt.scatter(data_outch192['Comp'],(data_outch192['MAC_original']-data_outch192['MAC'])/data_outch192['MAC_original']*100,label='Out_ch=192', color='c')
# plt.plot(xp192, p192(xp192), 'c' + '-')
# plt.scatter(data_outch256['Comp'], (data_outch256['MAC_original']-data_outch256['MAC'])/data_outch256['MAC_original']*100,label='Out_ch=256', color='m')
# plt.plot(xp256, p256(xp256), 'm' + '-')
# plt.scatter(data_outch320['Comp'], (data_outch320['MAC_original']-data_outch320['MAC'])/data_outch320['MAC_original']*100,label='Out_ch=320', color='k')
# plt.plot(xp320, p320(xp320), 'k' + '-')
# plt.scatter(data_outch384['Comp'], (data_outch384['MAC_original']-data_outch384['MAC'])/data_outch384['MAC_original']*100,label='Out_ch=384', color='y')
# plt.plot(xp384, p384(xp384), 'y' + '-')
# plt.legend()
# plt.xlabel('Compression')
# plt.ylabel('MACs decreased (%)')
# plt.title('MACs decreased (%) for Different Number of Output Channels')


# #%%

# data_tt=total_df_outch[total_df_outch['Dec']=='tt']

# data_outch192=data_tt[data_tt['Out_ch']==192]
# data_bas192=data_bas[data_bas['Out_ch']==192]
# data_outch256=data_tt[data_tt['Out_ch']==256]
# data_bas256=data_bas[data_bas['Out_ch']==256]
# data_outch320=data_tt[data_tt['Out_ch']==320]
# data_bas320=data_bas[data_bas['Out_ch']==320]
# data_outch384=data_tt[data_tt['Out_ch']==384]
# data_bas384=data_bas[data_bas['Out_ch']==384]

# data_outch192.loc[:, 'energy(kWh)'] = pd.to_numeric(data_outch192['energy(kWh)'], errors='coerce')
# data_outch192['delta_energy(kWh)']=np.ones(len(data_outch192['energy(kWh)']))*data_bas192['energy(kWh)'].to_numpy()-data_outch192['energy(kWh)']
# data_outch192.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_outch192['delta_energy(kWh)'], errors='coerce')
# data_outch192['per_energy(kWh)']=data_outch192['delta_energy(kWh)']/(np.ones(len(data_outch192['energy(kWh)']))*data_bas192['energy(kWh)'].to_numpy())*100
# data_outch192.loc[:,'per_energy(kWh)']=pd.to_numeric(data_outch192.loc[:,'per_energy(kWh)'],errors='coerce')
# data_outch192.loc[:,'per_energy(kWh)']=np.around(data_outch192.loc[:,'per_energy(kWh)'], decimals=2)
# z192 = np.polyfit(data_outch192['Comp'], data_outch192['delta_energy(kWh)'], 2)
# p192 = np.poly1d(z192)
# xp192 = np.linspace(0.1, 0.9, 100)


# data_outch256.loc[:, 'energy(kWh)'] = pd.to_numeric(data_outch256['energy(kWh)'], errors='coerce')
# data_outch256['delta_energy(kWh)']=np.ones(len(data_outch256['energy(kWh)']))*data_bas256['energy(kWh)'].to_numpy()-data_outch256['energy(kWh)']
# data_outch256.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_outch256['delta_energy(kWh)'], errors='coerce')
# data_outch256['per_energy(kWh)']=data_outch256['delta_energy(kWh)']/(np.ones(len(data_outch256['energy(kWh)']))*data_bas256['energy(kWh)'].to_numpy())*100
# data_outch256.loc[:,'per_energy(kWh)']=pd.to_numeric(data_outch256.loc[:,'per_energy(kWh)'],errors='coerce')
# data_outch256.loc[:,'per_energy(kWh)']=np.around(data_outch256.loc[:,'per_energy(kWh)'], decimals=2)

# z256 = np.polyfit(data_outch256['Comp'], data_outch256['delta_energy(kWh)'], 2)
# p256 = np.poly1d(z256)
# xp256 = np.linspace(0.1, 0.9, 100)

# data_outch320.loc[:, 'energy(kWh)'] = pd.to_numeric(data_outch320['energy(kWh)'], errors='coerce')
# data_outch320['delta_energy(kWh)']=np.ones(len(data_outch320['energy(kWh)']))*data_bas320['energy(kWh)'].to_numpy()-data_outch320['energy(kWh)']
# data_outch320.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_outch320['delta_energy(kWh)'], errors='coerce')
# data_outch320['per_energy(kWh)']=data_outch320['delta_energy(kWh)']/(np.ones(len(data_outch320['energy(kWh)']))*data_bas320['energy(kWh)'].to_numpy())*100
# data_outch320.loc[:,'per_energy(kWh)']=pd.to_numeric(data_outch320.loc[:,'per_energy(kWh)'],errors='coerce')
# data_outch320.loc[:,'per_energy(kWh)']=np.around(data_outch320.loc[:,'per_energy(kWh)'], decimals=2)

# z320 = np.polyfit(data_outch320['Comp'], data_outch320['delta_energy(kWh)'], 2)
# p320 = np.poly1d(z320)
# xp320 = np.linspace(0.1, 0.9, 100)

# data_outch384.loc[:, 'energy(kWh)'] = pd.to_numeric(data_outch384['energy(kWh)'], errors='coerce')
# data_outch384['delta_energy(kWh)']=np.ones(len(data_outch384['energy(kWh)']))*data_bas384['energy(kWh)'].to_numpy()-data_outch384['energy(kWh)']
# data_outch384.loc[:,'delta_energy(kWh)']=pd.to_numeric(data_outch384['delta_energy(kWh)'], errors='coerce')
# data_outch384['per_energy(kWh)']=data_outch384['delta_energy(kWh)']/(np.ones(len(data_outch384['energy(kWh)']))*data_bas384['energy(kWh)'].to_numpy())*100
# data_outch384.loc[:,'per_energy(kWh)']=pd.to_numeric(data_outch384.loc[:,'per_energy(kWh)'],errors='coerce')
# data_outch384.loc[:,'per_energy(kWh)']=np.around(data_outch384.loc[:,'per_energy(kWh)'], decimals=2)

# z384 = np.polyfit(data_outch384['Comp'], data_outch384['delta_energy(kWh)'], 2)
# p384 = np.poly1d(z384)
# xp384 = np.linspace(0.1, 0.9, 100)

# plt.figure()
# plt.scatter(data_outch192['Comp'], data_outch192['delta_energy(kWh)'],label='Out_ch=192', color='c')
# plt.plot(xp192, p192(xp192), 'c' + '-')
# plt.scatter(data_outch256['Comp'], data_outch256['delta_energy(kWh)'],label='Out_ch=256', color='m')
# plt.plot(xp256, p256(xp256), 'm' + '-')
# plt.scatter(data_outch320['Comp'], data_outch320['delta_energy(kWh)'],label='Out_ch=320', color='k')
# plt.plot(xp320, p320(xp320), 'k' + '-')
# plt.scatter(data_outch384['Comp'], data_outch384['delta_energy(kWh)'],label='Out_ch=384',color='y')
# plt.plot(xp384, p384(xp384), 'y' + '-')
# plt.title('Energy saved per number of input channels for cp')
# plt.xlabel('Compression')
# plt.ylabel('Energy saved [%]')
# plt.legend()
# # plt.figure()
# # plt.scatter(data_outch192['Comp'], data_outch192['energy(kWh)'],label='Out_ch=192')
# # plt.scatter(data_outch256['Comp'], data_outch256['energy(kWh)'],label='Out_ch=256')
# # plt.scatter(data_outch320['Comp'], data_outch320['energy(kWh)'],label='Out_ch=320')
# # plt.scatter(data_outch384['Comp'], data_outch384['energy(kWh)'],label='Out_ch=384')
# # plt.legend()
# # plt.xlabel('Compression')
# # plt.ylabel('Energy cost (kWh)')
# # plt.title('Energy Cost for Different Number of Output Channels')

# # plt.figure()
# # plt.scatter(data_outch192['Comp'], data_outch192['delta_energy(kWh)'],label='Out_ch=192')
# # plt.scatter(data_outch256['Comp'], data_outch256['delta_energy(kWh)'],label='Out_ch=256')
# # plt.scatter(data_outch320['Comp'], data_outch320['delta_energy(kWh)'],label='Out_ch=320')
# # plt.scatter(data_outch384['Comp'], data_outch384['delta_energy(kWh)'],label='Out_ch=384')
# # plt.legend()
# # plt.xlabel('Compression')
# # plt.ylabel('Energy savings(kWh)')
# # plt.title('Energy savings (kWh) for Different Number of Output Channels')

save_path = "outall_reg.pkl"
with open(save_path, 'wb') as f:
    pickle.dump(datanew, f)