# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:14:13 2024

@author: demib
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import seaborn as sns  # Import Seaborn for color palette
import scipy
import pickle

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
# # Set the font to use LaTeX
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })
plt.close('all')
pd.options.mode.chained_assignment = None  # default='warn'

#retrieve the Watt meter data
dataset=pd.read_csv('Voltcraft SEM5000_(202406)_power.csv')
dataset.columns = dataset.columns.str.strip()

#Remove the Voltage and Ampere measurments, and keep the power measurements
data=dataset[dataset.columns[0:3]]
#Change the Datetime stamp to be in the same format as the Python logger file
data['Datetime']=data['Date']+ ' ' + data['Time']
data['Datetime']=pd.to_datetime(data['Datetime'], format='%d-%m-%Y %H:%M')

#Combina date and time under same variable DateTime
data_struct=data[data.columns[1:4]]
data_struct=data_struct[['Datetime', 'Time', 'Power (W)']]
data_struct['Datetime'] = data_struct['Datetime'].apply(lambda x: x.replace(year=2024))

#retrieve the measurements logs
testlog=pd.read_csv('testen.csv')


def get_dicts(testlog,decompose):  
    testlog=testlog
    # Create an empty lists for the periods

    periods = {}
    # Find the different start and end pairs, by iterating over all the log rows
    for index, row in testlog.iterrows():
        
        if row['Info'].startswith(f'{decompose}-start'): #find a row that starts with start
            #read out the DateTime stamp
            start_time = pd.to_datetime(row['Datetime'])
            # Find the end time corresponding to the start time
            end_time = pd.to_datetime(testlog.loc[index + 1, 'Datetime'])    
            end_time_round=end_time.replace(second=0)
            
            #filter the period out of the watt meter measurements between the start and the end time
            close_strt = data_struct.iloc[(data_struct['Datetime'] - start_time).abs().argsort()[:1]]
            period = data_struct[(data_struct['Datetime'] >=close_strt.iloc[0]['Datetime']) & (data_struct['Datetime'] < end_time_round)]
               
            #Extract the name of the measurement presented after 'start'
            exp_name = row['Info'].split('start-')[1]  # Extract the text following 'start-'
            
            # Add filtered data with period name as key to dictionary
            periods[exp_name] = period
            
            #Add the extra seconds, since measurements are per minute
            periods[exp_name]['extra_t']=(end_time-end_time_round).total_seconds()

    # Create a new dictionary to find different runs of the same measurement
    all_runs = {}
    
    # Now find different runs for the same measurement by iteratinf over all period dict items
    for exp_name, runs in periods.items():

        # Extract the name of the experiment by removing the ind
        name = exp_name.split('-ind')[0]

        # Check if the type of measurement was alreadt seen in the list of periods
        if name in all_runs:
            # If so append the new run of that experiment
            all_runs[name].append(runs)
        else:
            # If not start a new list under the name of the experiment
            all_runs[name] = [runs] 
  
    # Create a new dictionary for the final results
    experiment_results = {}
    
    # Find all measurements for each experiment
    for experiment_name, experiment_runs in all_runs.items():
        # Create lists to store means, standard deviations, period lengths, and total energies
        means = []
        stds = []
        period_lengths = []
        total_energies = []
        total_energy_kwh=[]
        # Iterate over each different experiment, over the different runs
        for run in experiment_runs:
            # Calculate median over the power measurments
            power=run['Power (W)'].to_numpy()
            #To be sure check whether length of exp is larger than one, if so remove the first minute 
            if len(power)>1:
                median=np.median(power[1:])
            else:
                median=np.median(power)
            
            # Calculate period length, combining the seconds after full minutes
            period_length =len(power)-1+np.mean(run['extra_t']) / 60  # Convert to minutes
            
            # Calculate the totale energy based on the measured power and period length
            total_energy=median*period_length*60
            period_lengths.append(period_length)
            total_energies.append(total_energy)
            total_energy_kwh.append(total_energy/(3600*1000))
            
        # Calculate mean and standard deviation of total energies
        mean_total_energy = np.mean(total_energies)
        std_total_energy = np.std(total_energies)
        
        energy_kwh=np.mean(total_energy_kwh)
        energy_std=np.std(total_energy_kwh)
       
        # Add results to the experiment_results dictionary
        experiment_results[experiment_name] = {
            'period_lengths': period_lengths,
            'total_energies': total_energies,
            'total_energies_kwh':total_energy_kwh,
            'mean_total_energy': mean_total_energy,
            'std_total_energy': std_total_energy,
            'energy(kWh)': energy_kwh,
            'std_energy(kWh)': energy_std
        }
    return experiment_results, all_runs, periods

#get baseline results
bas_results, bas_runs, bas_periods=get_dicts(testlog, 'bas')

#get tt results
tt_results, tt_runs, tt_periods=get_dicts(testlog, 'dec')


save_path = "data_tt_dec_inch.pkl"
with open(save_path, 'wb') as f:
    pickle.dump(tt_results, f)
save_path = "data_bas_inch.pkl"
with open(save_path, 'wb') as f:
    pickle.dump(bas_results, f)    


#%%


runs1=tt_runs['outch512-inch384-factcp-r0.1-wh4']
runs2=tt_runs['outch512-inch320-facttt-r0.5-wh4']
runs3=tt_runs['outch512-inch256-facttucker-r0.9-wh4']
runs4=tt_runs['outch512-inch192-factcp-r0.25-wh4']
runs5=tt_runs['outch512-inch256-facttt-r0.75-wh4']
runs6=tt_runs['outch512-inch384-facttucker-r0.1-wh4']

plt.figure()
for i in [1, 2, 3]:
    run = runs1[i - 1]
    plt.plot(range(len(run['Power (W)'])), run['Power (W)'], label=f'Inch384 cp0.1 Run {i}')
for i in [1, 2, 3]:
    run = runs2[i - 1]
    plt.plot(range(len(run['Power (W)'])), run['Power (W)'], label=f'Inch320 tt0.5 Run {i}')
for i in [1, 2, 3]:
    run = runs3[i - 1]
    plt.plot(range(len(run['Power (W)'])), run['Power (W)'], label=f'Inch256 tucker0.9 Run {i}')
for i in [1, 2, 3]:
    run = runs4[i - 1]
    plt.plot(range(len(run['Power (W)'])), run['Power (W)'], label=f'Inch192 cp0.25 Run {i}')
for i in [1, 2, 3]:
    run = runs5[i - 1]
    plt.plot(range(len(run['Power (W)'])), run['Power (W)'], label=f'Inch256 tt0.75 Run {i}')
for i in [1, 2, 3]:
    run = runs6[i - 1]
    plt.plot(range(len(run['Power (W)'])), run['Power (W)'], label=f'Inch384 tucker0.1 Run {i}')

plt.ylabel('Power [W]')
plt.xlabel('Time [min]')
plt.title('Runs for different measurements on the CPU')
plt.xlim([0, 18])
plt.tight_layout()

plt.show()