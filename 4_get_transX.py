#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 17:26:04 2018

@author: xuweijia
"""
import json
spa=0.9
train_file='data/final_train/train_tokenized_spa{}.json'.format(spa)
transX_dir='transX_spa{}'.format(spa)
#transX_dir='transX'
#train_file='data/final_train/train_tokenized.json'
with open(train_file,'r') as f:
    train_samples=json.load(f)

r_set=set()
e_set=set() # 58617
# train
train_set=set() # triple
for sample in train_samples:
    e1_id,p_id,e2_id=sample['triple'][0]
    e_set.add(e1_id)
    e_set.add(e2_id)
    r_set.add(p_id)
    train_set.add((e1_id,p_id,e2_id))  # only id that matters

R=len(r_set)
E=len(e_set)
r_list=list(r_set)
e_list=list(e_set)

r_dict=dict(zip(r_list,range(R)))
e_dict=dict(zip(e_list,range(E)))   # eid 2 transx idx
train_list=list(train_set)
import subprocess
subprocess.call(['mkdir', '-p', transX_dir]) 

with open(transX_dir+'/e_id2idx.json','w') as f:    # Qid to transX idx
    json.dump(e_dict,f)
with open(transX_dir+'/r_id2idx.json','w') as f:    # Pid to transX idx
    json.dump(r_dict,f)
    
f_train=open(transX_dir+'/train2id.txt','w')
f_e=open(transX_dir+'/entity2id.txt','w')
f_r=open(transX_dir+'/relation2id.txt','w')
# sample
f_test=open(transX_dir+'/test2id.txt','w')
f_valid=open(transX_dir+'/valid2id.txt','w')
# predict
f_test_orig=open(transX_dir+'/original_test2id.txt','w')
f_valid_orig=open(transX_dir+'/original_valid2id.txt','w')

# train e1,e2,r
f_train.write(str(len(train_set))+'\n') 
for e1,p,e2 in train_list:
    #e1='_'.join(sample['e1'].split())
    e1_id=e_dict[e1]
    #e2='_'.join(sample['answear'].split())
    e2_id=e_dict[e2]
    #r='_'.join(sample['query'].split())
    r_id=r_dict[p]
    f_train.write(str(e1_id)+'\t'+str(e2_id)+'\t'+str(r_id)+'\n')
f_train.close()


# Qid idx
f_e.write(str(E)+'\n')
for e in e_list:# original label 
    f_e.write(e+'\t'+str(e_dict[e])+'\n')                      # wiki  eid  -->  idx
    #f_e.write('_'.join(e.split()) +'\t'+str(e_train_id_dict[e])+'\n')  # yoga  eid==elabel  -->  idx
f_e.close()

# Pid idx
f_r.write(str(R)+'\n')
for r in r_list:
    f_r.write(r+'\t'+str(r_dict[r])+'\n')                           # wiki
    #f_r.write('_'.join(r.split())+'\t'+str(r_id_dict[r])+'\n')        # yoga  rid==rlabel  -->  idx
f_r.close()



file='data/final_test/dev_contain_e.json'
with open(file,'r') as f:
    samples=json.load(f)
l=0
for sample in samples:
    l+=len(sample['ans_id'])

# valid
f_valid.write(str(l)+'\n')         # 2846 indepent triple
f_valid_orig.write(str(len(samples))+'\n')      # 2190  left 609 orig ans_list
for sample in samples:
    
    e1=sample['e1_id']
    e1_id=e_dict[e1]
    r=sample['p_id']
    r_id=r_dict[r]
    
    f_valid_orig.write(str(e1_id) + '\t')  #e1
    f_valid_orig.write(str(r_id))          #r
    for e2 in sample['ans_id']:
        e2_id=e_dict[e2]
        f_valid.write(str(e1_id)+'\t'+str(e2_id)+'\t'+str(r_id)+'\n')
        f_valid_orig.write('\t'+str(e2_id)) #e1
    f_valid_orig.write('\n')

f_valid.close()
f_valid_orig.close()
print(len(samples),l)

# test
file='data/final_test/test_contain_e.json'
with open(file,'r') as f:
    samples=json.load(f)
l=0
for sample in samples:
    l+=len(sample['ans_id'])
f_test.write(str(l)+'\n')         # 2846
f_test_orig.write(str(len(samples))+'\n')      # 2190  left 609
for sample in samples:
    e1=sample['e1_id']
    e1_id=e_dict[e1]
    r=sample['p_id']
    r_id=r_dict[r]
    
    f_test_orig.write(str(e1_id) + '\t')  #e1
    f_test_orig.write(str(r_id))          #r
    for e2 in sample['ans_id']:
        e2_id=e_dict[e2]
        f_test.write(str(e1_id)+'\t'+str(e2_id)+'\t'+str(r_id)+'\n')
        f_test_orig.write('\t'+str(e2_id)) #e1
    f_test_orig.write('\n')

f_test.close()
f_test_orig.close()
print(len(samples),l)

print({'E':E},{'T':len(train_set)},{'R':R})
#185000
#4421 4937
#4454 4881
#{'E': 45128} {'T': 67637} {'R': 78}

# 0.6
#4421 4937
#4454 4881
#{'E': 45128} {'T': 40583} {'R': 78}

# 0.7
#4421 4937
#4454 4881
#{'E': 45128} {'T': 47346} {'R': 78}

# 0.8
#4421 4937
#4454 4881
#{'E': 45128} {'T': 54110} {'R': 78}

# 0.9
#4421 4937
#4454 4881
#{'E': 45128} {'T': 60874} {'R': 78}