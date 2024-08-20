import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import matplotlib.patches as mpatches

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
# RGB values from the image
colors = [
    '#4477aa',  # Blue
    '#66ccee',  # Light Blue
    '#228833',  # Green
    '#ccbb44',  # Yellow
    '#ee6677',  # Red
    '#aa3377',  # Pink
    '#bbbbbb'   # Grey
]

color1 = colors[0]
color2 = colors[2]
color3 = colors[4]

greenalg1 = np.array([11.31, 14.14, 14.14, 14.14, 14.14]) / 1000
greenalg2 = np.array([19.8, 36.77, 39.60, 42.42, 42.42]) / 1000
greenalg3 = np.array([11.31, 22.63, 39.6, 70.71, 84.85]) / 1000

cbtr1 = np.array([13.56, 24.16, 24.4, 25.8, 25.62]) / 10000
cbtr2 = np.array([12.58, 42.6, 78.9, 82.69, 82.82]) / 10000
cbtr3 = np.array([18.71, 44.77, 96.57, 150.65, 186.66]) / 10000

act1 = np.array([10.39, 22.06, 23.26, 24.69, 25.05]) / 10000
act2 = np.array([39.30, 78.70, 82.03, 85.43, 86.35]) / 10000
act3 = np.array([16.39, 47.51, 103.21, 170.82, 205.33]) / 10000

c = [0.1, 0.25, 0.5, 0.75, 0.9]

plt.figure()
a1,=plt.plot(c, greenalg1, color=color1, marker='*')
a2,=plt.plot(c, greenalg2, color=color1, marker='o')
a3,=plt.plot(c, greenalg3, color=color1, marker='D')
b1,=plt.plot(c, cbtr1, color=color2, marker='*')
b2,=plt.plot(c, cbtr2, color=color2, marker='o')
b3,=plt.plot(c, cbtr3, color=color2, marker='D')
c1,=plt.plot(c, act1, color=color3, marker='*')
c2,=plt.plot(c, act2, color=color3, marker='o')
c3,=plt.plot(c, act3, color=color3, marker='D')

# # Define the labels for the custom legend
# legend_labels = {
#     'Green algorithms': color1,  # Blue
#     'Carbontracker': color2,     # Green
#     'Watt meter': color3         # Red
# }

# # Define markers
# markers = ['o', 'D', '*']
# measurements = ['Meas. 1', 'Meas. 2', 'Meas. 3']

# # Create custom legend entries
# legend_entries = []

# # Add measurement legends with colors grouped
# for meas, marker in zip(measurements, markers):
#     legend_entries.append(mlines.Line2D([1], [1], color=color1, marker=marker, linestyle='None', markersize=8, label=meas))
#     legend_entries.append(mlines.Line2D([2], [2], color=color2, marker=marker, linestyle='None', markersize=8))
#     legend_entries.append(mlines.Line2D([], [], color=color3, marker=marker, linestyle='None', markersize=8))

# # Add the custom legend
# plt.legend(handles=[legend_entries[],], loc='upper left', ncol=3, title='Measurements and Methods')

# # Add title and labels
plt.title('Energy Consumption for Different Tools')
plt.xlabel('Compression Ratio')
plt.ylabel('Energy (kWh)')
first_legend=plt.legend([(a1,b1,c1),(a2,b2,c2),(a3,b3,c3)],['Exp 1','Exp 2', 'Exp 3'], numpoints=1,
              handler_map={tuple: HandlerTuple(ndivide=3)})
patches = [
    mpatches.Patch(color=color1, label='Green algorithms'),
    mpatches.Patch(color=color2, label='Carbontracker'),
    mpatches.Patch(color=color3, label='Watt meter')
]


plt.gca().add_artist(first_legend)
plt.legend(handles=patches, loc='upper center')

# Save the plot as an SVG file
svg_filename = 'C:/Users/demib/Documents/Thesis/Data_final_processing/Figures/Discussion/tools.pdf'
plt.savefig(svg_filename, format='pdf')



plt.show()

