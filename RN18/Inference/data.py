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
# # Set the font to use LaTeX
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })
plt.close('all')
pd.options.mode.chained_assignment = None  # default='warn'

dataset=pd.read_csv('Voltcraft SEM5000_(202406)_power.csv')
dataset.columns = dataset.columns.str.strip()

data=dataset[dataset.columns[0:3]]
data['Datetime']=data['Date']+ ' ' + data['Time']
data['Datetime']=pd.to_datetime(data['Datetime'], format='%d-%m-%Y %H:%M')

data_struct=data[data.columns[1:4]]
data_struct=data_struct[['Datetime', 'Time', 'Power (W)']]
data_struct['Datetime'] = data_struct['Datetime'].apply(lambda x: x.replace(year=2024))
#data_struct.loc[:, 'Datetime'] += pd.Timedelta(hours=1)

testlog=pd.read_csv('testen.csv')

# Create an empty list to store periods
periods_dict = {}

# Iterate over each pair of start and end times
for index, row in testlog.iterrows():
    if row['Info'].startswith('start'):
        start_time = pd.to_datetime(row['Datetime'])
        # Find the end time corresponding to the start time
        end_time = pd.to_datetime(testlog.loc[index + 1, 'Datetime'])    
        end_time_round=end_time.replace(second=0)
        #end_time1=end_time-timedelta(minutes=1)
            
        close_strt = data_struct.iloc[(data_struct['Datetime'] - start_time).abs().argsort()[:1]]
        #close_end = data_struct.iloc[(data_struct['Datetime'] - end_time).abs().argsort()[:1]]
        filtered_period = data_struct[(data_struct['Datetime'] >=close_strt.iloc[0]['Datetime']) & (data_struct['Datetime'] < end_time_round)]
        
        # Calculate central tendency measure (e.g., mean or median)
        #mean_val = filtered_period['Power (W)'].mean()  # You can use median() instead of mean() if needed
        
        # Filter out small outliers
        #filtered_period = filtered_period[filtered_period['Power (W)'] >= 0.8*mean_val]  # Adjust the threshold as needed
        
       # Extract the name of the period from the Info column
        period_name = row['Info'].split('start-')[1]  # Extract the text following 'start-'
        
        # Add filtered data with period name as key to dictionary
        periods_dict[period_name] = filtered_period
        periods_dict[period_name]['extra_t']=(end_time-end_time_round).total_seconds()


# Create a new dictionary to store aggregated dataframes
runs_data_dict = {}

# Iterate over each key-value pair in the filtered data dictionary
for period_name, filtered_df in periods_dict.items():
    # Extract the common prefix
    prefix = period_name.split('-ind')[0]
    # Check if the prefix already exists in the aggregated data dictionary
    if prefix in runs_data_dict:
        # If it exists, append the dataframe to the list of dataframes under the prefix
        runs_data_dict[prefix].append(filtered_df)
    else:
        # If it doesn't exist, create a new list with the dataframe under the prefix
        runs_data_dict[prefix] = [filtered_df]

dataset_run=pd.DataFrame(runs_data_dict)


#dataset_fin.to_pickle('dataset_fn.pkl')


# Create an empty dictionary to store results
experiment_results = {}

# Iterate over each experiment in the dictionary
for experiment_name, experiment_dataframes in runs_data_dict.items():
    # Initialize lists to store means, standard deviations, period lengths, and total energies
    means = []
    stds = []
    period_lengths = []
    total_energies = []
    total_energy_kwh=[]
    # Iterate over each dataframe for the experiment
    for dataframe in experiment_dataframes:
        # Calculate mean and standard deviation of power values
        power=dataframe['Power (W)'].to_numpy()
        median=np.median(power[1:])
        #power=power[4:]
        # Calculate period length
        period_length =len(power)+np.mean(dataframe['extra_t']) / 60  # Convert to minutes
        period_length=period_length
        # Append results to lists
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

dataset_fin=pd.DataFrame(experiment_results)
# dataset_fin=dataset_interm.loc[['energy(kWh)', 'std_energy(kWh)', 'mean_total_energy', 'std_total_energy']]

dataset_fin.to_pickle('dataset_final.pkl')
# #%%
# colors = sns.color_palette("pastel")

# # # Get total energies for each experiment
# experiments = ['bascifar10-mdlrn18','dec-cp-r0.1-lay[10]', 'dec-cp-r0.1-lay[10, 8]', 'dec-cp-r0.1-lay[10, 8, 6, 4]']
# total_energies = [experiment_results[exp]['energy(kWh)'] for exp in experiments]
# error=[experiment_results[exp]['std_energy(kWh)'] for exp in experiments]
# plt.figure(figsize=(6,6))

# bar_width = 0.5
# index = range(len(experiments))

# # Plot bars for total energy and idle consumption with overlap
# plt.bar(index, total_energies, width=bar_width, color=colors[0], label='Measured Energy')
# plt.bar(index, [14 * 60 / (3600 * 1000)] * len(experiments), width=bar_width, color=colors[1], label='Idle Consumption')

# # Plot error bars
# plt.errorbar(index, total_energies, yerr=error, fmt='o', color='black', capsize=5)

# plt.xlabel('Experiments')
# plt.ylabel('Energy (kWh)')
# plt.ylim(0,0.02)
# plt.title('Energy (kWh) on Garipov (Cifar10) for different # of layers')

# plt.xticks(index, experiments, rotation=45, ha='right')
# plt.legend(['Measurement','Idle consumpt','Errorbar'],loc='lower left')
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# plt.tight_layout()
# plt.show()



# #%%
# colors = sns.color_palette("pastel")

# # # Get total energies for each experiment
# experiments = ['dec-cp-r0.1-lay[54]','dec-cp-r0.2-lay[54]', 'dec-cp-r0.4-lay[54]', 'dec-cp-r0.5-lay[54]', 'dec-cp-r0.6-lay[54]', 'dec-cp-r0.7-lay[54]', 'dec-cp-r0.8-lay[54]']
# total_energies = [experiment_results[exp]['energy(kWh)'] for exp in experiments]
# error=[experiment_results[exp]['std_energy(kWh)'] for exp in experiments]
# plt.figure(figsize=(6, 6))

# bar_width = 0.5
# index = range(len(experiments))

# # Plot bars for total energy and idle consumption with overlap
# plt.bar(index, total_energies, width=bar_width, color=colors[0], label='Measured Energy')
# plt.bar(index, [14 * 60 / (3600 * 1000)] * len(experiments), width=bar_width, color=colors[1], label='Idle Consumption')

# # Plot error bars
# plt.errorbar(index, total_energies, yerr=error, fmt='o', color='black', capsize=5)

# plt.xlabel('Experiments')
# plt.ylabel('Energy (kWh)')
# plt.ylim(0,0.075)
# plt.title('Energy (kWh) on Rn18 (Cifar10) for different # of layers')

# plt.xticks(index, experiments, rotation=45, ha='right')
# plt.legend(['Measurement','Idle consumpt','Errorbar'],loc='lower left')
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# plt.tight_layout()
# plt.show()

#%% Plot cp,tt, tucker for layer 63
# Define color palette
# colors = sns.color_palette("pastel")

# experiments = ['bascifar10-mdlrn18', 'dec-cp-r0.1-lay[63]', 'dec-cp-r0.1-lay[54]']
# total_energies = [experiment_results[exp]['energy(kWh)'] for exp in experiments]
# error = [experiment_results[exp]['std_energy(kWh)'] for exp in experiments]

# plt.figure(figsize=(6, 6))

# bar_width = 0.5
# index = range(len(experiments))

# # Plot bars for total energy and idle consumption with overlap
# plt.bar(index, total_energies, width=bar_width, color=colors[0], label='Measured Energy')
# plt.bar(index, [14 * 60 / (3600 * 1000)] * len(experiments), width=bar_width, color=colors[1], label='Idle Consumption')

# # Plot error bars
# plt.errorbar(index, total_energies, yerr=error, fmt='o', color='black', capsize=5)

# plt.xlabel('Experiments')
# plt.ylabel('Energy (kWh)')
# plt.ylim(0,0.05)
# plt.title('Energy (kWh) on Rn18 (Cifar10) for different # of layers')

# plt.xticks(index, experiments, rotation=45, ha='right')
# plt.legend(['Measurement','Idle consumpt','Errorbar'],loc='lower left')

# plt.grid(axis='y', linestyle='--', alpha=0.7)

# plt.tight_layout()
# plt.show()


# #%%plot cp, tucker, tt for layer 54
# experiments = ['bascifar10-mdlrn18','dec-cp-r0.1-lay[54]', 'dec-tucker-r0.1-lay[54]', 'dec-tt-r0.1-lay[54]']
# total_energies = [experiment_results[exp]['energy(kWh)'] for exp in experiments]
# error=[experiment_results[exp]['std_energy(kWh)'] for exp in experiments]

# plt.figure(figsize=(6, 6))

# bar_width = 0.5
# index = range(len(experiments))

# # Plot bars for total energy and idle consumption with overlap
# plt.bar(index, total_energies, width=bar_width, color=colors[0], label='Measured Energy')
# plt.bar(index, [14 * 60 / (3600 * 1000)] * len(experiments), width=bar_width, color=colors[1], label='Idle Consumption')

# # Plot error bars
# plt.errorbar(index, total_energies, yerr=error, fmt='o', color='black', capsize=5)

# plt.xlabel('Experiments')
# plt.ylabel('Energy (kWh)')
# plt.ylim(0,0.075)
# plt.title('Energy (kWh) on Rn18 (Cifar10) for different # of layers')

# plt.xticks(index, experiments, rotation=45, ha='right')
# plt.legend(['Measurement','Idle consumpt','Errorbar'],loc='lower left')
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# plt.tight_layout()
# plt.show()


# #%%cp, tucker, tt for different layers

# # Define color palette
# colors = sns.color_palette("pastel")

# # Original experiment names
# experiments = ['dec-cp-r0.1-lay[63]', 'dec-tucker-r0.1-lay[63]', 'dec-tt-r0.1-lay[63]']
# experiments2 = ['dec-cp-r0.1-lay[54]', 'dec-tucker-r0.1-lay[54]', 'dec-tt-r0.1-lay[54]']

# # Abbreviated experiment names
# abbrev_experiments = ['dec-cp-r0.1', 'dec-tucker-r0.1', 'dec-tt-r0.1']

# total_energies = [experiment_results[exp]['energy(kWh)'] for exp in experiments]
# error = [experiment_results[exp]['std_energy(kWh)'] for exp in experiments]

# total_energies2 = [experiment_results[exp]['energy(kWh)'] for exp in experiments2]
# error2 = [experiment_results[exp]['std_energy(kWh)'] for exp in experiments2]

# # Single bar data
# bascifar10_energy = experiment_results['bascifar10-mdlrn18']['energy(kWh)']
# error3=experiment_results['bascifar10-mdlrn18']['std_energy(kWh)']

# plt.figure(figsize=(6, 6))

# bar_width = 0.25  # Reduced bar width
# index = range(1, len(experiments) + 1)  # Start from 1 for other bars

# # Plot bars for total energy
# plt.bar(index, total_energies, width=bar_width, color=colors[0], label='Measured Energy')
# # Plot bars for idle consumption
# plt.bar([i + bar_width for i in index], total_energies2,  width=bar_width, color=colors[5], label='Idle Consumption')

# # Plot single bar for bascifar10-mdlrn18 data
# plt.bar(0.5, bascifar10_energy, width=bar_width, color=colors[2], label='bascifar10-mdlrn18')  # Adjusted x position

# # Plot error bars
# plt.errorbar(index, total_energies, yerr=error, fmt='o', color='black', capsize=5)
# plt.errorbar([i + bar_width for i in index], total_energies2, yerr=error2, fmt='o', color='black', capsize=5)
# plt.errorbar(0.5, bascifar10_energy, yerr=error3, fmt='o', color='black', capsize=5)


# plt.xlabel('Experiments')
# plt.ylabel('Energy (kWh)')
# plt.title('Energy (kWh) on Rn18 (Cifar10) for different # of layers')

# # Change x-axis ticks to include bascifar10-mdlrn18
# plt.xticks([0.5] + [i + bar_width / 2 for i in index], ['bascifar10-mdlrn18'] + abbrev_experiments, rotation=45, ha='right')
# plt.legend(['Layer 63', 'Layer 54','No compression', 'Errorbar'], loc='lower left')
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# plt.tight_layout()
# plt.show()



# #%%
# period=runs_data_dict['dec-cp-r0.1-lay[63]'][0]['Power (W)']
# period1=runs_data_dict['dec-cp-r0.1-lay[63]'][1]['Power (W)']
# period2=runs_data_dict['dec-cp-r0.1-lay[63]'][2]['Power (W)']
# period3=runs_data_dict['dec-cp-r0.1-lay[63]'][3]['Power (W)']
# period4=runs_data_dict['dec-cp-r0.1-lay[63]'][4]['Power (W)']

# period5=runs_data_dict['bascifar10-mdlrn18'][0]['Power (W)']
# period6=runs_data_dict['bascifar10-mdlrn18'][1]['Power (W)']
# period7=runs_data_dict['bascifar10-mdlrn18'][2]['Power (W)']
# period8=runs_data_dict['bascifar10-mdlrn18'][3]['Power (W)']
# period9=runs_data_dict['bascifar10-mdlrn18'][4]['Power (W)']


# t=9
# plt.figure()
# plt.plot(range(t),period, 'g')
# plt.plot(range(t),period1, 'g')
# plt.plot(range(t),period2, 'g')
# plt.plot(range(t),period3,'g')
# plt.plot(range(t),period4, 'g')
# plt.plot(range(t),period5,  'b')
# plt.plot(range(t),period6,'b')
# plt.plot(range(t),period7,'b')
# plt.plot(range(t),period8,'b')
# plt.plot(range(t),period9, 'b')

# #%%
# period=runs_data_dict['dec-cp-r0.1-lay[41]'][0]['Power (W)']
# period1=runs_data_dict['dec-cp-r0.1-lay[41]'][1]['Power (W)']
# period2=runs_data_dict['dec-cp-r0.1-lay[41]'][2]['Power (W)']
# period3=runs_data_dict['dec-cp-r0.1-lay[41]'][3]['Power (W)']
# period4=runs_data_dict['dec-cp-r0.1-lay[41]'][4]['Power (W)']

# t=9
# plt.figure()
# plt.plot(range(t),period, 'b', label='runs')
# plt.plot(range(t),period1, 'b')
# plt.plot(range(t),period2, 'b')
# plt.plot(range(t),period3, 'b')
# plt.plot(range(t),period4, 'b')
# plt.ylim([200,320])
# plt.title('Power (W) measured at each minute')
# plt.xlabel('Time (min)')
# plt.ylabel('Power (W)')
# plt.legend()