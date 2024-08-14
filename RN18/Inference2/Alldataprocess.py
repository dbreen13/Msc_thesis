import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Custom sort key functions
def custom_sort_key(s):
    parts = s.split('-')
    compression_type = parts[1]
    r_value = float(parts[2][1:])
    lay_value = int(parts[3][4:-1])
    compression_order = {'cp': 0, 'tucker': 1, 'tt': 2}
    return (compression_order[compression_type], -lay_value, r_value)

def mem_custom_sort_key(s):
    parts = s.split('-')
    compression_type = parts[0]
    r_value = float(parts[1][1:])
    lay_value = int(parts[2][4:-1])
    compression_order = {'cp': 0, 'tucker': 1, 'tt': 2}
    return (compression_order[compression_type], -lay_value, r_value)

# Create custom colormap function
def create_custom_colormap():
    colors = [(0.0, "#a00000"), (0.2, "#c46666"), (0.4, "#d8a6a6"), 
              (0.5, "white"), 
              (0.6, "white"), (0.8, "#3594cc"), (1.0, "#2066a8")]
    n_bins = 100  # Discretizes the interpolation into bins
    cmap_name = 'custom_diverging'
    return LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# Preprocess data function
def preprocess_data(data, baseline):
    data.loc[:, 'energy(kWh)'] = pd.to_numeric(data['energy(kWh)'], errors='coerce')
    data['delta_energy(kWh)'] = baseline['energy(kWh)'].to_numpy() - data['energy(kWh)']
    data.loc[:, 'delta_energy(kWh)'] = pd.to_numeric(data['delta_energy(kWh)'], errors='coerce')
    data['per_energy(kWh)'] = data['delta_energy(kWh)'] / baseline['energy(kWh)'].to_numpy() * 100
    data.loc[:, 'per_energy(kWh)'] = pd.to_numeric(data['per_energy(kWh)'], errors='coerce')
    data.loc[:, 'per_energy(kWh)'] = np.around(data['per_energy(kWh)'], decimals=2)
    return data

# Create heatmap data function
def create_heatmap_data(data, pivot_index, pivot_columns, pivot_values):
    return data.pivot(pivot_index, pivot_columns, pivot_values)

# Plot combined heatmaps function with single color bar
def plot_combined_heatmaps(heatmap_data_list, titles, xlabel, ylabel, cmap, vmin, vmax):
    fig, axes = plt.subplots(1, 4, figsize=(24, 8), gridspec_kw={'width_ratios': [1, 1, 1, 0.05]})

    for i, (heatmap_data, title) in enumerate(zip(heatmap_data_list, titles)):
        sns.heatmap(heatmap_data, annot=False, fmt=".1e", cmap=cmap, cbar=False,
                    vmin=vmin, vmax=vmax, annot_kws={"size": 12}, ax=axes[i])
        axes[i].set_title(title, pad=20, fontsize=12)
        axes[i].set_xlabel(xlabel, fontsize=12)
        axes[i].set_ylabel(ylabel, fontsize=12)
        axes[i].tick_params(axis='both', which='major', labelsize=12)
        axes[i].grid(False)
        
    # Create single color bar
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)), cax=axes[-1], orientation='vertical')
    cbar.set_label('Energy Savings (kWh)', fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.grid(False)
    
    pdf_filename = 'C:/Users/demib/Documents/Thesis/Data_final_processing/Figures/RN18.pdf'
    plt.savefig(pdf_filename, format='pdf')
    plt.show()

# Load data
data = pd.read_pickle('allinfo.pkl')
data_eng = pd.read_pickle('dataset_final.pkl').transpose()

data_eng = data_eng[:-1]

# Sort the index using the custom sort key
sorted_index = sorted(data_eng.index, key=custom_sort_key)
data_eng2 = data_eng.loc[sorted_index]

# Load memory measurements
mem = pd.read_pickle('mem_rn18.pkl')
mem = mem.rename(columns={'Mem': 'Mem_meas'})
sorted_index = sorted(mem.index, key=mem_custom_sort_key)
mem = mem.loc[sorted_index]
mem.index = data_eng2.index

baseline=pd.read_pickle('base.pkl')
baseline = baseline.to_frame().transpose()

mem_bas = pd.read_pickle('mem_bas_rn18.pkl')
mem_bas.index = baseline.index
baseline = pd.concat([baseline, mem_bas], axis=1)
membas_tot = pd.concat([mem_bas] * 270, ignore_index=True)

data.reset_index(drop=True, inplace=True)
data_eng2.reset_index(drop=True, inplace=True)
mem.reset_index(drop=True, inplace=True)

# Make final dataset
data = pd.concat([data_eng2, data, mem], axis=1)
data['MAC_red'] = data['MAC_original'] - data['MAC']
data['Mem_meas_diff'] = (data['Mem_meas'].to_numpy() - membas_tot.to_numpy())[0, :]

# Separate data per method
maskcp = data['Dec'].isin(['cp'])
datacp = data.copy()[maskcp]

masktt = data['Dec'].isin(['tt'])
datatt = data.copy()[masktt]

masktucker = data['Dec'].isin(['tucker'])
datatucker = data.copy()[masktucker]

# Create colormap
cmap = create_custom_colormap()

# Preprocess data
datacp = preprocess_data(datacp, baseline)
datatt = preprocess_data(datatt, baseline)
datatucker = preprocess_data(datatucker, baseline)

# Create heatmap data
heatmap_data_cp = create_heatmap_data(datacp, "Comp", "Layer", "per_energy(kWh)")
heatmap_data_tuck = create_heatmap_data(datatucker, "Comp", "Layer", "per_energy(kWh)")
heatmap_data_tt = create_heatmap_data(datatt, "Comp", "Layer", "per_energy(kWh)")

# Concatenate all data
all_data = pd.concat([datacp, datatt, datatucker])

# Calculate vmin and vmax
vmin = all_data['per_energy(kWh)'].min()
vmax = np.abs(vmin)

# Plot combined heatmaps
titles = [
    'Tucker Decomposition',
    'CP Decomposition',
    'TT Decomposition'
]
plot_combined_heatmaps([heatmap_data_tuck, heatmap_data_cp, heatmap_data_tt], titles, 'Layer', 'Compression Ratio', cmap, vmin, vmax)
