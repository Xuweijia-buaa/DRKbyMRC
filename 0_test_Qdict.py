#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 21:35:35 2018

@author: xuweijia
"""
import json

# get all Q in train_file
Q_set=set()
def return_Q(input_file):
    global Q_set
    with open(input_file,'r') as f:
        samples = json.load(f) 
    for sample in samples:
        for c in sample['candis_links'].values():
            Q_list=[e[0] for e in c]
            Q_set|=set(Q_list)
            
def return_test_Q(input_file):
    global Q_set
    with open(input_file,'r') as f:
        samples = json.load(f) 
    for sample in samples:
        for  can_dict in sample['candis_links']:
            for c in can_dict.values():
                Q_list=[e[0] for e in c]
                Q_set|=set(Q_list)
       
return_Q('data/train_before_statis.json')
return_test_Q('/media/xuweijia/06CE5DF1CE5DD98F/useful_old_data/new_NER_data/e_30/up_30/dev_contain_e.json')
return_test_Q('/media/xuweijia/06CE5DF1CE5DD98F/useful_old_data/new_NER_data/e_30/up_30/test_contain_e.json')
return_test_Q('/media/xuweijia/06CE5DF1CE5DD98F/useful_old_data/new_NER_data/e_30/up_30/dev_no_e.json')
return_test_Q('/media/xuweijia/06CE5DF1CE5DD98F/useful_old_data/new_NER_data/e_30/up_30/test_no_e.json')

new_dict=dict()        
with open('/media/xuweijia/06CE5DF1CE5DD98F/useful_old_data/new_NER_data/e_30/Q_id_dsp_30.json','r') as f:
    Q_id_dep=json.load(f)
    n=0
    for Q_id in Q_id_dep:
        if Q_id in Q_set:
            new_dict[Q_id]=Q_id_dep[Q_id]
            n+=1
            print('n:{}'.format(n))

with open('data/Q_desp.json','w') as f:
    json.dump(new_dict,f)
    
