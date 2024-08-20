#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:28:49 2024

@author: dbreen
"""
import ast

import pickle
import numpy as np
import pandas as pd
from tensorly import validate_tt_rank
from tensorly import validate_cp_rank
from tensorly import validate_tucker_rank



batch=128
in_features=[(batch,64,16,16),(batch,128,8,8), (batch,256,4,4), (batch,512,2,2)]
out_features=[(batch,64,16,16),(batch,128,8,8), (batch,256,4,4), (batch,512,2,2)]

layers=[63,57,51,47,41,35,28,25,19,6]    
compression=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

#%%MACS for decompsing
df = pd.DataFrame()
for method in ['cp', 'tucker', 'tt']:   
    for layer in layers: 
        for comp in compression:      
            stride=1
            if layer==6 or layer==15 or layer==9 or layer==25 or layer==19:
                in_feature=in_features[0]
                if layer==19 or layer==25:
                    out_feature=out_features[1]
                    stride=2
                else:
                    out_feature=out_features[0]
            elif layer==22 or layer==28 or layer==35 or layer==41:
                in_feature=in_features[1]
                if layer==28 or layer==22:
                    out_feature=out_features[1]
                else:
                    out_feature=out_features[2]
                    stride=2
            elif layer==38 or layer==47 or layer==51 or layer==57:
                in_feature=in_features[2]
                if layer==38 or layer==47:
                    out_feature=out_features[2]
                else:
                    out_feature=out_features[3]
                    stride=2
            else:
                in_feature=in_features[3]
                out_feature=out_features[3]
            if layer==25 or layer==41 or layer==57:
                kernel=1
            else: 
                kernel=3
            d=kernel
            S=in_feature[1]
            T=out_feature[1]
            W=in_feature[2]
            if method=='tucker':
                R=validate_tucker_rank(tensor_shape=(out_feature[1],in_feature[1],kernel,kernel),rank=comp)

                R1=R[1]
                R2=R[2]
                R3=R[3]
                R4=R[0]
                macs=batch*(W**2*S*R2+2*(R1*R2*R3*R4)+R1*R2*(W/stride)**2*d**2+R1*T*(W/stride**2))
                mem=batch*W**2*R2+batch*R1*(W/stride)**2+2*R1*R2*R3*R4
                membas=mem+T*(W/stride)**2+R1*R2*R3*R4+T*R1+S*R2+d*R3+d*R3           
            elif method=='tt':
                R=validate_tt_rank(tensor_shape=(out_feature[1],in_feature[1],kernel,kernel),rank=comp)

                R1=R[2]
                R2=R[3]
                R3=R[1]
                macs=batch*(W**2*S*R1+R2*(R1*d*W*W/stride+d*R3*(W/stride)**2)+(W/stride)**2*T*R3)
                mem=batch*R1*W**2+batch*R2*W*W/stride+batch*R3*(W/stride)**2
                membas=mem+T*(W/stride)**2+S*R1+R1*d*R2+R2*d*R3+T*R3
            elif method=='cp':
                R=validate_cp_rank(tensor_shape=(out_feature[1],in_feature[1],kernel,kernel),rank=comp)

                macs=batch*(W**2*R*S+R*d*W*W/stride+R*d*(W/stride)**2+(W/stride)**2*T*R)
                mem=batch*(R*W**2+R*W/stride*W+R*(W/stride)**2)
                membas=mem+T*(W/stride)**2+T*R+S*R+d*R*2
            string=f'dec-{method}-r{comp}-lay[{layer}]'
            df1 = pd.DataFrame({'Method': string,'Layer':layer,'MAC': macs,'Rank':[np.array(R)],'Memcalc_diff':mem, 'Memcalc_tot':membas, 'Comp':comp, 'Stride':stride, 'Kernel':kernel, 'In_ch':in_feature[1], 'Out_ch':out_feature[1]}, index=[0])
            df=pd.concat([df,df1], ignore_index=True)
                        
#%%MACS before decomposing      

df_original = pd.DataFrame()
for method in ['cp', 'tucker', 'tt']:   
    for layer in layers: 
        for comp in compression: 
            if layer==6 or layer==15 or layer==9 or layer==25 or layer==19:
                in_feature=in_features[0]
                if layer==19 or layer==25:
                    out_feature=out_features[1]
                else:
                    out_feature=out_features[0]
            elif layer==22 or layer==28 or layer==35 or layer==41:
                in_feature=in_features[1]
                if layer==28 or layer==22:
                    out_feature=out_features[1]
                else:
                    out_feature=out_features[2]
            elif layer==38 or layer==47 or layer==51 or layer==57:
                in_feature=in_features[2]
                if layer==38 or layer==47:
                    out_feature=out_features[2]
                else:
                    out_feature=out_features[3]
            else:
                in_feature=in_features[3]
                out_feature=out_features[3]
            if layer==25 or layer==41 or layer==57:
                kernel=1
            else: 
                kernel=3   
        
            macs=in_feature[1]*out_feature[1]*kernel**2*out_feature[3]**2*batch
            string=f'dec-{method}-r{comp}-lay[{layer}]'
            df1 = pd.DataFrame({'Method': string, 'MAC_original': macs, 'In_feat':[np.array(in_feature)], 'Out_feat': [np.array(out_feature)], 'Dec': method}, index=[0])
            df_original=pd.concat([df_original,df1], ignore_index=True)
            
#%%Combine dataframes

dataset_final=pd.concat([df['MAC'], df['Memcalc_diff'],df['Rank'], df['Comp'],df['Layer'],df_original['MAC_original'],df_original['In_feat'],df_original['Dec'],df_original['Out_feat'], df['Out_ch'], df['In_ch'], df['Stride'], df['Kernel']], axis=1)

save_path = "allinfo.pkl"
with open(save_path, 'wb') as f:
    pickle.dump(dataset_final, f)