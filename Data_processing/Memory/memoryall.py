# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 16:00:17 2024

@author: demib
"""

import pandas as pd


#Laptop
mem_feat=pd.read_pickle('toy_problems/Features/mem_feat.pkl')
bas_feat=mem_feat[:4]
mem_feat=mem_feat[4:]

mem_kern=pd.read_pickle('toy_problems/Kernel/mem_kern.pkl')
bas_kern=pd.read_pickle('toy_problems/mem_bas_kern.pkl')

mem_in=pd.read_pickle('toy_problems/In_ch/mem_inch.pkl')
bas_in=pd.read_pickle('toy_problems/membas_inch')

mem_out=pd.read_pickle('toy_problems/Out_ch/mem_outch.pkl')
bas_out=pd.read_pickle('')