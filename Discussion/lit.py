import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Update ggplot styles
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
    'font.size': 12,
}

plt.rcParams.update(ggplot_styles)

# Load data
littot = pd.read_pickle('lit_in.pkl')
antot = pd.read_pickle('allinfo_in.pkl')

# Filter data for CP
litcp = littot[littot['Dec'] == 'cp']
ancp = antot[antot['Dec'] == 'cp']
litin192_cp = litcp[litcp['In_ch'] == 192]
anin192_cp = ancp[ancp['In_ch'] == 192]
litmac_cp = litin192_cp[['MAC', 'Comp']]
anmac_cp = anin192_cp[['MAC', 'Comp']]

# Filter data for Tucker
littucker = littot[littot['Dec'] == 'tt']
antucker = antot[antot['Dec'] == 'tt']
litin192_tucker = littucker[littucker['In_ch'] == 192]
anin192_tucker = antucker[antucker['In_ch'] == 192]
litmac_tucker = litin192_tucker[['MAC', 'Comp']]
anmac_tucker = anin192_tucker[['MAC', 'Comp']]

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Plot for CP
axs[0].plot(litmac_cp['Comp'], litmac_cp['MAC'], label='MACs from literature')
axs[0].plot(anmac_cp['Comp'], anmac_cp['MAC'], label='MACs from thesis')
axs[0].set_ylabel('# of MAC operations')
axs[0].set_xlabel('Compression')
axs[0].legend()
axs[0].set_title('The MAC operations for CP')

# Plot for Tucker
axs[1].plot(litmac_tucker['Comp'], litmac_tucker['MAC'], label='MACs from literature')
axs[1].plot(anmac_tucker['Comp'], anmac_tucker['MAC'], label='MACs from thesis')
axs[1].set_ylabel('# of MAC operations')
axs[1].set_xlabel('Compression')
axs[1].legend()
axs[1].set_title('The MAC operations for TT')

# Adjust layout and show plot
plt.tight_layout()
plt.show()
