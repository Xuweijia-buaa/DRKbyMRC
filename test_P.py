#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 21:29:16 2018

@author: xuweijia
"""

P_dir='/media/xuweijia/06CE5DF1CE5DD98F/useful_old_data/new_NER_data/sorted_p_with_label_delete.txt'
with open(P_dir) as f:
    P=f.readlines()
P_id=[p.strip().split('\t')[0] for p in P]
P_label=[p.strip().split('\t')[1] for p in P]
P_dict=dict(zip(P_id,P_label))

count_P_n=dict(zip(P_dict.keys(),[0 for i in P_dict.keys()] ))
count_P_n_=dict(zip(P_dict.keys(),[0 for i in P_dict.keys()] ))