# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:15:54 2024

@author: demib
"""
import pandas as pd
import matplotlib.pyplot as plt


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


df=pd.read_pickle('mem_feat.pkl')
base=df[:4]
df=df[4:]
info=pd.read_pickle('allinfo_feat.pkl')


sort_keys=sorted(df.index, key=custom_sort_key)
df = df.loc[sort_keys]

# Create the new index based on the specified format
new_index = [
    f"outch{row['Out_ch']}-inch{row['In_ch']}-fact{row['Dec']}-r{row['Comp']}-wh{row['In_feat'][2]}"
    for idx, row in info.iterrows()
]
info.index=new_index

data_tot=pd.concat([df,info], axis=1)

#%%
data_cp=data_tot[data_tot['Dec']=='cp']
data_cp = data_cp[data_cp['In_feat'].apply(lambda x: x.tolist() == [128, 448, 4,4])]

data_tt=data_tot[data_tot['Dec']=='tt']
data_tt = data_tt[data_tt['In_feat'].apply(lambda x: x.tolist() == [128, 448, 4,4])]

data_tuck=data_tot[data_tot['Dec']=='tucker']
data_tuck = data_tuck[data_tuck['In_feat'].apply(lambda x: x.tolist() == [128, 448, 4,4])]

#%%

plt.figure()
plt.scatter(data_cp['Comp'], data_cp['Mem'],label='cp', color='c')
plt.scatter(data_tuck['Comp'], data_tuck['Mem'],label='tuck', color='y')
plt.scatter(data_tt['Comp'], data_tt['Mem'],label='tt', color='m')
plt.legend()
plt.xlabel('Compression')
plt.ylabel('Memory (Mb)))')
plt.title('Memory per method')
