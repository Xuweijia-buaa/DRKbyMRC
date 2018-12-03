#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:17:42 2018

@author: xuweijia
"""
import json
import numpy as np
# just test final used train_file
#train_file='/media/xuweijia/06CE5DF1CE5DD98F/useful_old_data/new_NER_data/e_30/train_20w_list.json'
train_file='/home/xuweijia/my_drqa/data/train_before_statis_new.json'
with open(train_file,'r') as f:
    samples=json.load(f)
    
#with open('/media/xuweijia/06CE5DF1CE5DD98F/useful_old_data/new_NER_data/e_30/train_20w_list.json','r') as f:
#    samples=json.load(f)

n_anchor_notin=0                       # anchor not all in original
n_valid_notin=0                        # valid not all in original
n_c=0                                  # all mention
n_c_entity=0
n_a=0                                  # ans mention
n_a_entity=0

n_mul_m=0
new_list=[]
count_p=dict()                       # each p's answear
count_p_count=dict()                 # each p's answear's linked entity number
t_set=set()
e_set=set()
for n,sample in enumerate(samples):
    
    e1_id,p_id,e2_id=sample['triple']         # sample['ans_list']=t_labels   sample['ans_id']=t_ids 
    e_set.add(e1_id)
    e_set.add(e2_id)
    t_set.add((e1_id,p_id,e2_id))
    # test how many anchor not in NER  
    true_Q=list(sample['Q2anchor'].values())
    n_not_in=False
    if len(true_Q)!=0:
        for c in true_Q:
            if c not in sample['orig_cands']:
                n_not_in=True
                break
    if n_not_in:
        n_anchor_notin+=1
        
    # test how many valid not in NER         
    n_not_in=False
    for c in sample['valid_cands']:
        if c not in sample['orig_cands']:
            n_not_in=True
            break
    if n_not_in:
        n_valid_notin+=1  

    # compute avrage entity each mention linked
    for c in sample['valid_cands']:                   # sample['candis']
        if len(sample['candis_links'][c])!=1:
            n_c+=1
            n_c_entity+=len(sample['candis_links'][c])

    # find surface mention for each ans:
    mention=sample['ans_mention']
    if count_p.get(p_id)!=None:
       count_p[p_id]+=1
       count_p_count[p_id]+=len(sample['candis_links'][mention])
    else:
       count_p[p_id]=1
       count_p_count[p_id]=len(sample['candis_links'][mention]) 
    
    print("n:{}".format(n))
    
        # new_list.append(sample)
        # ans_mention's linked candidate
    n_a+=1
    n_a_entity+=len(sample['candis_links'][sample['ans_mention']])
        
    # new_list.append(sample)
        
P_dir='/media/xuweijia/06CE5DF1CE5DD98F/useful_old_data/new_NER_data/sorted_p_with_label_delete.txt'
with open(P_dir,'r') as f:
    P=f.readlines()
P_id=[p.strip().split('\t')[0] for p in P]
P_label=[p.strip().split('\t')[1] for p in P]
P_dict=dict(zip(P_id,P_label))

new_dict=dict()
for p_id, p_time in count_p.items():
    v=count_p_count[p_id]/p_time
    new_dict[P_dict[p_id]]= v         
        
    
avrage_entity=n_c_entity/n_c                   # avrage entity each mention linked
avrage_ans_entity=n_a_entity/n_a               # avrage entity each ans mention linked
anchor_fil_rate=n_anchor_notin/len(samples)    # how many ahchor can't be NER
valid_fil_rate=n_valid_notin/len(samples)      # how many valid candidas can't be NER

print("avrage entity each mention linked:" ,avrage_entity)
print("avrage entity each ans mention linked:" ,avrage_ans_entity)
print("how many ahchor can't be NER:", anchor_fil_rate)
print("how many valid candidas can't be NER:",valid_fil_rate)
print('n_e',len(e_set),'n_t',len(t_set))

with open('data/dict_p.json','w') as f:
    json.dump(new_dict,f)
    
# short
#avrage entity each mention linked: 4.274634254704706
#avrage entity each ans mention linked: 1.240681229636431
#how many ahchor can't be NER: 0.4743758846290364
#how many valid candidas can't be NER: 0.4743758846290364
#how many samples man have multiple answear: 0.016844651472592384
    
# long
#avrage entity each mention linked: 4.311297005707138
#avrage entity each ans mention linked: 1.2412364236490556
#how many ahchor can't be NER: 0.4745669073832919
#how many valid candidas can't be NER: 0.4745669073832919
#how many samples man have multiple answear: 0.016986087543681958     3874 samples have mentions more than one candidates

# long mention change into anchor
#avrage entity each mention linked: 4.311297005707138
#avrage entity each ans mention linked: 1.2385298535651226
#how many ahchor can't be NER: 0.4745669073832919
#how many valid candidas can't be NER: 0.4745669073832919
#how many samples man have multiple answear: 0.0017143934511047096
    
# long mention (del doc not contain both)
#avrage entity each mention linked: 3.9686051949156145
#avrage entity each ans mention linked: 1.137703039452862
#how many ahchor can't be NER: 0.47021481710533497
#how many valid candidas can't be NER: 0.47021481710533497
    
# long mention new (del doc not contain both)
#avrage entity each mention linked: 3.969480352709528
#avrage entity each ans mention linked: 1.138322047410041
#how many ahchor can't be NER: 0.466229279869517
#how many valid candidas can't be NER: 0.466206372041454
    
# long mention new (after token replace)
#avrage entity each mention linked: 3.935775426657451
#avrage entity each ans mention linked: 1.1467473769168683
#how many ahchor can't be NER: 0.3794296475652408
#how many valid candidas can't be NER: 0.3787516814635459
#n_e 45146 n_t 67806  (original n_e 51039 n_t 79935)