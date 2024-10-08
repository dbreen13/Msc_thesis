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
    feat = int(parts[4][2:])
    
    # Assign a numeric value to compression types to ensure correct order
    compression_order = {'factcp': 0, 'facttucker': 1, 'facttt': 2}
    
    # Return a tuple that will be used for sorting
    return (compression_order[compression_type],r_value, feat )


def process_kernel_data(total_df, data_bas, dec_method, kernel_sizes):

    # Filter data for the specific decomposition method
    data_tt = total_df[total_df['Dec'] == dec_method]

    # Initialize an empty list to store processed data
    processed_data = []

    for kernel in kernel_sizes:
        # Filter data for the specific kernel size
        data_kern = data_tt[data_tt['Kernel'] == kernel]
        bas_kern = data_bas[data_bas['Kern'] == kernel]
        
        # Convert energy(kWh) to numeric
        data_kern.loc[:, 'energy(kWh)'] = pd.to_numeric(data_kern['energy(kWh)'], errors='coerce')

        # Calculate delta energy
        data_kern['delta_energy(kWh)'] = np.ones(len(data_kern['energy(kWh)'])) * bas_kern['energy(kWh)'].to_numpy() - data_kern['energy(kWh)']
        data_kern.loc[:, 'delta_energy(kWh)'] = pd.to_numeric(data_kern['delta_energy(kWh)'], errors='coerce')

        # Calculate percentage energy
        data_kern['per_energy(kWh)'] = data_kern['delta_energy(kWh)'] / (np.ones(len(data_kern['energy(kWh)'])) * bas_kern['energy(kWh)'].to_numpy()) * 100
        data_kern.loc[:, 'per_energy(kWh)'] = pd.to_numeric(data_kern.loc[:, 'per_energy(kWh)'], errors='coerce')
        data_kern.loc[:, 'per_energy(kWh)'] = np.around(data_kern.loc[:, 'per_energy(kWh)'], decimals=2)

        # Append the processed data to the list
        processed_data.append(data_kern)

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


#%%Combine all datasets to one larger one
allinfo_kernel=pd.read_pickle('allinfo_kern.pkl')

data_tot=pd.DataFrame.from_dict(pd.read_pickle('data_tt_dec_kernel.pkl')).transpose()

data_bas=pd.DataFrame.from_dict(pd.read_pickle('data_bas_kernel.pkl')).transpose()
data_bas['Kern'] = data_bas.index.to_series().str.extract(r'kern(\d+)')[0].astype(float)

sort_keys=sorted(data_tot.index, key=custom_sort_key)
data_tot = data_tot.loc[sort_keys]

# Create the new index based on the specified format so it is similar to the other ones
new_index = [
    f"outch{row['Out_ch']}-inch{row['In_ch']}-fact{row['Dec']}-r{row['Comp']}-wh{row['In_feat'][2]}-kern{row['Kernel']}"
    for idx, row in allinfo_kernel.iterrows()
]

# Assign the new index to the DataFrame
allinfo_kernel.index = new_index

#load memory measurements
mem_kern=pd.read_pickle('mem_kern (1).pkl')
mem_kern=mem_kern.rename(columns={'Mem':'Mem_meas'})
mem_kern.index=new_index
mem_kern=mem_kern

mem_bas=pd.read_pickle('mem_bas_kern.pkl')

membas_tot=pd.concat([mem_bas] * 15, ignore_index=True)



total_df_outch=pd.concat([data_tot,allinfo_kernel, mem_kern], axis=1)
total_df_outch['Mem_meas_diff']=total_df_outch['Mem_meas']-membas_tot['Mem'].to_numpy()

save_path = "kernall_reg.pkl"
with open(save_path, 'wb') as f:
    pickle.dump(total_df_outch, f)




#%%


kernel_sizes = [1, 3, 5]
data_tucker_new = process_kernel_data(total_df_outch, data_bas, 'tucker',kernel_sizes)
data_cp_new = process_kernel_data(total_df_outch, data_bas, 'cp',kernel_sizes)
data_tt_new = process_kernel_data(total_df_outch, data_bas, 'tt',kernel_sizes)

datanew=pd.concat([data_cp_new, data_tucker_new, data_tt_new])
datanew=datanew.rename(columns={'Kernel':'Kern'})

# Merge datanew with data_bas to get baseline std_energy
datanew['std_energy(kWh)'] = datanew['std_energy(kWh)'].astype(float)
data_bas['std_energy(kWh)'] = data_bas['std_energy(kWh)'].astype(float)

datanew = datanew.merge(data_bas[['std_energy(kWh)','Kern']], on='Kern', suffixes=('', '_bas'))
# Calculate combined standard deviation
datanew['combined_std_energy'] = np.sqrt(datanew['std_energy(kWh)']**2 + datanew['std_energy(kWh)_bas']**2)

kernels = [1, 3, 5]
colors = ['#4477aa', '#66ccee', '#228833']

# Set up the figure and subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
methods = datanew['Dec'].unique()

# Plot data for each decomposition method in a separate subplot
for ax, method in zip(axes, methods):
    method_data = datanew[datanew['Dec'] == method]
    for color, kernel in zip(colors, kernels):
        subset = method_data[method_data['Kern'] == kernel]
        # Plot the line
        sns.lineplot(
            data=subset,
            x='Comp',
            y='delta_energy(kWh)',
            ax=ax,
            label=f'Kernel={kernel}',
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
            plt.legend(title='Kernel size')
    ax.set_title(f'{title} Decomposition')
    ax.set_xlabel('Compression Ratio')
axes[0].set_ylabel('Energy saved (kWh)')

# Add a main title
plt.suptitle('Energy Consumption Across Compression Ratios by Kernel Sizes and Decomposition Methods', y=0.98)

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 0.99, 1])

# Save the plot as a PDF file
plt.savefig('C:/Users/demib/Documents/Thesis/Data_final_processing/Figures/Desktop/dp_eng_kernel.pdf', format='pdf')

# Show the plot
plt.show()

#%%
compression_ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
kernels = [1, 3, 5]

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
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, gridspec_kw={'wspace': 0.3})

# Define y-limits for each column
ylim_macs = [-2.8e9, 2.55e9]
ylim_memory = [0, 420]
ylim_memory_diff = [0, 1.8e7]

def plot_method(ax, method_data, method_name, colors, compression_ratios, kernels, ylim=None):
    bar_width = 0.1
    x = np.arange(len(kernels))
    for i, cr in enumerate(compression_ratios):
        offset = (i - 2) * (bar_width + 0.01)  # Adjust the offset for each bar within a group
        ax.bar(x + offset, method_data[cr], width=bar_width, color=colors[cr], label=f'{cr}', zorder=2)
    ax.set_title(method_name)
    ax.set_xticks(x)
    ax.set_xticklabels(kernels)
    ax.grid(True, which='both', zorder=1)
    if ylim:
        ax.set_ylim(ylim)

# # Plot MAC reduction data for each method

plot_method(axes[0], cp_memory_meas, 'CP', colors, compression_ratios, kernels, ylim=ylim_memory)
plot_method(axes[1], tucker_memory_meas, 'Tucker', colors, compression_ratios, kernels, ylim=ylim_memory)
plot_method(axes[2], tt_memory_meas, 'TT', colors, compression_ratios, kernels, ylim=ylim_memory)


fig.text(0.02, 0.5, 'Memory (MB)', va='center', rotation='vertical', fontsize=12)
plt.subplots_adjust(left=0.06, right=0.98, top=0.87, bottom=0.1, wspace=0.3, hspace=0.3)


fig.text(0.5, 0.005, 'Kernels', ha='center', fontsize=12)

# Add legend
handles, labels = axes[0].get_legend_handles_labels()

#plt.suptitle('Experiment 3: Memory and Computation Complexity', x=0.5, y=0.98, ha='center', fontsize=14, fontweight='bold')
plt.suptitle('Measured Memory for Experiment 4 on GPU')
plt.legend(title='Compression ratio')
# Save the plot as a PDF file
pdf_filename = 'C:/Users/demib/Documents/Thesis/Data_final_processing/Figures/Desktop/dp_mem_kernel.pdf'
plt.savefig(pdf_filename, format='pdf')

plt.show()


save_path = "kernall_reg.pkl"
with open(save_path, 'wb') as f:
    pickle.dump(datanew, f)