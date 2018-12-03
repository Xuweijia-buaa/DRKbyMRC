#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 09:42:58 2018

@author: xuweijia
"""

import subprocess
import json
import copy
final_train_dir='data/final_train/'  
with open(final_train_dir+'train_tokenized.json','r') as f:
    train_samples=json.load(f)
r_set=set()
e_set=set()
train_set=set() # triple
for sample in train_samples:
    e1_id,p_id,e2_id=sample['triple'][0]
    e_set.add(e1_id)
    e_set.add(e2_id)
    r_set.add(p_id)
    train_set.add((e1_id,p_id,e2_id))

import os
def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path                                       # 本身是文件，返回文件名（生成器）
    elif os.path.isdir(path):                            # 文件夹
        for dirpath, _, filenames in os.walk(path):      # dirpath:代表目录的路径; 包含了dirpath下所有子目录的名字;  filenames：非目录文件的名字
            for f in filenames:
                yield os.path.join(dirpath, f)           # 绝对路径：目录+文件，返回所有文件
    else:
        raise RuntimeError('Path %s is invalid' % path)

final_test_dir='data/final_test' 
subprocess.call(['mkdir', '-p', final_test_dir]) 
##P_dict_file='data/dict_p.json'
#P_dict_file='data/dict_p_{}_{}.json'
#with open(P_dict_file,'r') as f:
#    P_dict=json.load(f)
file_orig='data/dev_contain_e.json'
file_rep='data/dev_rep_contain_e.json'
file_token='data/dev_contain_e_valid_cands_tokenized_all.json'
file_rep_token='data/dev_contain_e_rep_valid_cands_tokenized_all.json'
with open(file_orig,'r') as f:
    samples=json.load(f)
with open(file_rep,'r') as f:
    samples_rep=json.load(f)
with open(file_token,'r') as f:
    samples_token=json.load(f)
with open(file_rep_token,'r') as f:
    samples_rep_token=json.load(f)
new_samples=[]
new_samples_rep=[]
new_samples_token=[]
new_samples_rep_token=[]
e_dev_set=copy.copy(e_set)
n_repeat=0
train_dev_set=copy.copy(train_set)
for ex_id,sample in enumerate(samples):
    e1_id=sample['e1_id']
    ans_ids=sample['ans_id']
    p_id=sample['p_id']
    triple=[e1_id,p_id,ans_ids]
    sample_rep=samples_rep[ex_id]
    sample_token=samples_token[ex_id]
    sample_rep_token=samples_rep_token[ex_id]
    
    assert triple==[sample_rep['e1_id'],sample_rep['p_id'],sample_rep['ans_id']]
    assert triple==sample_token['triple'][0]
    assert triple==sample_rep_token['triple'][0]
    
    flag_repeat=False
    for e2_id in ans_ids:
       if (e1_id,p_id,e2_id) in train_set:
           flag_repeat=True
           break
       
    if flag_repeat==True:  
         n_repeat+=1
         continue
    
    if e1_id in e_set and len([e2 in e_set for e2 in ans_ids])==len(ans_ids):
       new_samples.append(sample)
       new_samples_rep.append(sample_rep)
       new_samples_token.append(sample_token)
       new_samples_rep_token.append(sample_rep_token)
       e_dev_set.add(e1_id)
       for e2_id in ans_ids:
           e_dev_set.add(e1_id)
           train_dev_set.add((e1_id,p_id,e2_id))           
file_orig=final_test_dir+'/dev_contain_e.json'
file_rep=final_test_dir+'/dev_rep_contain_e.json'
file_token=final_test_dir+'/dev_contain_e_valid_cands_tokenized_all.json'
file_rep_token=final_test_dir+'/dev_contain_e_rep_valid_cands_tokenized_all.json'
with open(file_orig,'w') as f:
    json.dump(new_samples,f)
with open(file_rep,'w') as f:
    json.dump(new_samples_rep,f)
with open(file_token,'w') as f:
    json.dump(new_samples_token,f)
with open(file_rep_token,'w') as f:
    json.dump(new_samples_rep_token,f)    
       
n_dev_repeat=n_repeat
n_dev_old=samples
n_dev=len(new_samples) # 4421
       
# test
file_orig='data/test_contain_e.json'
file_rep='data/test_rep_contain_e.json'
file_token='data/test_contain_e_valid_cands_tokenized_all.json'
file_rep_token='data/test_contain_e_rep_valid_cands_tokenized_all.json'
with open(file_orig,'r') as f:
    samples=json.load(f)
with open(file_rep,'r') as f:
    samples_rep=json.load(f)
with open(file_token,'r') as f:
    samples_token=json.load(f)
with open(file_rep_token,'r') as f:
    samples_rep_token=json.load(f)
new_samples=[]
new_samples_rep=[]
new_samples_token=[]
new_samples_rep_token=[]
n_repeat=0
for ex_id,sample in enumerate(samples):
    e1_id=sample['e1_id']
    ans_ids=sample['ans_id']
    p_id=sample['p_id']
    triple=[e1_id,p_id,ans_ids]
    sample_rep=samples_rep[ex_id]
    sample_token=samples_token[ex_id]
    sample_rep_token=samples_rep_token[ex_id]
    
    assert triple==[sample_rep['e1_id'],sample_rep['p_id'],sample_rep['ans_id']]
    assert triple==sample_token['triple'][0]
    assert triple==sample_rep_token['triple'][0]
    
    flag_repeat=False
    for e2_id in ans_ids:
       if (e1_id,p_id,e2_id) in train_dev_set:
           flag_repeat=True
           break       
    if flag_repeat==True:  
         n_repeat+=1
         continue
    
    if e1_id in e_set and len([e2 in e_set for e2 in ans_ids])==len(ans_ids):
       new_samples.append(sample)
       new_samples_rep.append(sample_rep)
       new_samples_token.append(sample_token)
       new_samples_rep_token.append(sample_rep_token)
        
file_orig=final_test_dir+'/test_contain_e.json'
file_rep=final_test_dir+'/test_rep_contain_e.json'
file_token=final_test_dir+'/test_contain_e_valid_cands_tokenized_all.json'
file_rep_token=final_test_dir+'/test_contain_e_rep_valid_cands_tokenized_all.json'
with open(file_orig,'w') as f:
    json.dump(new_samples,f)
with open(file_rep,'w') as f:
    json.dump(new_samples_rep,f)
with open(file_token,'w') as f:
    json.dump(new_samples_token,f)
with open(file_rep_token,'w') as f:
    json.dump(new_samples_rep_token,f)    
       
n_test_repeat=n_repeat   
n_test_old=samples
n_test=len(new_samples) # 4454




       
       

        
