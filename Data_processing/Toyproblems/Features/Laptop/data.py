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


def get_dicts(testlog,decompose):  
    # Create an empty list to store periods
    testlog=testlog
    periods_dict = {}
    # Iterate over each pair of start and end times
    for index, row in testlog.iterrows():
        if row['Info'].startswith(f'{decompose}-start'):
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
            if len(power)>1: 
                median=np.median(power[1:])
            else:
                median=np.median(power)

            #power=power[4:]
            # Calculate period length
            period_length =len(power)-1+np.mean(dataframe['extra_t']) / 60  # Convert to minutes
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
    return experiment_results, runs_data_dict, periods_dict

#get baseline results
bas_results, bas_runs, bas_periods=get_dicts(testlog, 'bas')

#get tt results
tt_results, tt_runs, tt_periods=get_dicts(testlog, 'dec')



# save_path = "data_tt_dec_feat.pkl"
# with open(save_path, 'wb') as f:
#     pickle.dump(tt_results, f)
# save_path = "data_bas_feat.pkl"
# with open(save_path, 'wb') as f:
#     pickle.dump(bas_results, f)    


#%%
