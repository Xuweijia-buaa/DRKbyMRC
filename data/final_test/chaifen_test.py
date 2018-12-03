#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:44:11 2018

@author: xuweijia
"""
# test no_e
#import json
#final_train_dir='../final_train/'  
#with open(final_train_dir+'train_tokenized.json','r') as f:
#    train_samples=json.load(f)
#r_set=set()
#e_set=set()
#train_set=set() # triple
#for sample in train_samples:
#    e1_id,p_id,e2_id=sample['triple'][0]
#    e_set.add(e1_id)
#    e_set.add(e2_id)
#    r_set.add(p_id)
#    train_set.add((e1_id,p_id,e2_id))
#    
#no_e_file='../all_no_e_valid_cands_tokenized_all.json'
#with open(no_e_file,'r') as f:
#    samples=json.load(f)
#for sample in samples:
#    e1_id,p_id,e2_ids=sample['triple'][0]
#    for e2_id in e2_ids:
#        if e1_id in e_set and e2_id in e_set:
#            raise ValueError
#        if (e1_id,p_id,e2_id) in train_set:
#            raise ValueError
import json  
mode='test'
valid_can=True
contain=False
P_dict_file='dict_p_{}_{}.json'.format(mode,valid_can) if contain else 'dict_no_e.json'
with open(P_dict_file,'r') as f:
    P_dict=json.load(f)
if contain:
    file_token='{}_contain_e_valid_cands_tokenized_all.json'.format(mode)
    file_rep_token='{}_contain_e_rep_valid_cands_tokenized_all.json'.format(mode)
else:
    file_token='no_e_valid_cands_tokenized_all.json'
    file_rep_token='no_e_rep_valid_cands_tokenized_all.json'
with open(file_token,'r') as f:
    samples_token=json.load(f)
with open(file_rep_token,'r') as f:
    samples_rep_token=json.load(f)
    
gap=1.5
new_list_single=[]
new_list_multi=[]
new_list_single_rep=[]
new_list_multi_rep=[]
for ex_id,sample in enumerate(samples_token):
    sample_rep=samples_rep_token[ex_id]
    p_label=sample['triple'][1][1]
    if p_label in P_dict:
        if P_dict[p_label]>gap:
            new_list_multi.append(sample)
            new_list_multi_rep.append(sample_rep)
        else:
        # elif P_dict[p_label]<=1.05:
            new_list_single.append(sample)
            new_list_single_rep.append(sample_rep)
if contain:               
    file_single='{}_contain_e_valid_cands_tokenized_single.json'.format(mode)
    file_multi='{}_contain_e_valid_cands_tokenized_mul.json'.format(mode)
    file_rep_single='{}_contain_e_rep_valid_cands_tokenized_single.json'.format(mode)
    file_rep_multi='{}_contain_e_rep_valid_cands_tokenized_mul.json'.format(mode)
else:
    file_single='no_e_valid_cands_tokenized_single.json'
    file_multi='no_e_valid_cands_tokenized_mul.json'
    file_rep_single='no_e_rep_valid_cands_tokenized_single.json'
    file_rep_multi='no_e_rep_valid_cands_tokenized_mul.json'   
    
with open(file_single,'w') as f:
    json.dump(new_list_single,f)
with open(file_multi,'w') as f:
    json.dump(new_list_multi,f)
with open(file_rep_single,'w') as f:
    json.dump(new_list_single_rep,f)
with open(file_rep_multi,'w') as f:
    json.dump(new_list_multi_rep,f)
    
print({'n_multi':len(new_list_multi)},{'n_single':len(new_list_single)},{'total':len(samples_token)})

# dev:{'n_multi': 1427} {'n_single': 2985} {'total': 4421}
# test {'n_multi': 2620} {'n_single': 1830} {'total': 4454}
# no_e {'n_multi': 4729} {'n_single': 1591} {'total': 6320}