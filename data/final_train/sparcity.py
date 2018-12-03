#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 22:50:34 2018

@author: xuweijia
"""
from utils import spa
import json
train_orig='train.json'
with open(train_orig,'r') as f:
    samples=json.load(f)
    
train_token='train_tokenized.json'
with open(train_token,'r') as f:
    samples_token=json.load(f)
    
train_rep_token='train_rep_tokenized.json'
with open(train_rep_token,'r') as f:
    samples_rep=json.load(f)
    
e_set=set()
t_set=set()
for sample in samples_token:
    e1_id,p_id,e2_id=sample['triple'][0]
    e_set.add(e1_id)
    e_set.add(e2_id)
    t_set.add((e1_id,p_id,e2_id))

t_set=list(t_set)
T=len(t_set)   # 67637

import random
random.shuffle(t_set)
new_e_set=set()
t_remain=set()                       # those are must include t.  remain t, drop (1-)
for t_id,t in enumerate(t_set):
    e1_id,p_id,e2_id=t
    if e1_id not in new_e_set:
        t_remain.add(t)
        new_e_set.add(e1_id)
    if e2_id not in new_e_set:
        t_remain.add(t)
        new_e_set.add(e2_id)


# set parameter:  &
sparsity=0.9
spa(sparsity,T,t_remain,t_set,samples,samples_token,samples_rep,train_orig,train_token,train_rep_token)

sparsity=0.8
spa(sparsity,T,t_remain,t_set,samples,samples_token,samples_rep,train_orig,train_token,train_rep_token)

sparsity=0.7
spa(sparsity,T,t_remain,t_set,samples,samples_token,samples_rep,train_orig,train_token,train_rep_token)

sparsity=0.6
spa(sparsity,T,t_remain,t_set,samples,samples_token,samples_rep,train_orig,train_token,train_rep_token)




#{'sparsity': 0.9}
#{'t_new': 60874} {'t_old': 67637} {'rate:': 0.9000103493649926}
#{'n new': 185000} {'n orig': 164228} {'rate:': 0.887718918918919}
#{'sparsity': 0.8}
#{'t_new': 54110} {'t_old': 67637} {'rate:': 0.8000059139228529}
#{'n new': 185000} {'n orig': 143583} {'rate:': 0.7761243243243243}
#{'sparsity': 0.7}
#{'t_new': 47346} {'t_old': 67637} {'rate:': 0.7000014784807133}
#{'n new': 185000} {'n orig': 122486} {'rate:': 0.6620864864864865}
#{'sparsity': 0.6}
#{'t_new': 40583} {'t_old': 67637} {'rate:': 0.6000118278457057}
#{'n new': 185000} {'n orig': 100949} {'rate:': 0.5456702702702703}


#sparsity=0.9
#n_remain=int(T*sparsity)
#if len(t_remain)>n_remain:
#    print({'least sparsity':len(t_remain)/T})  # 0.550
#    print('enough, can not  be smaller')
#else:
#    t_id=0
#    while (len(t_remain)<=n_remain and t_id<len(t_set)):            # make t_remain len max to n_remain  
#        t=t_set[t_id]
#        if t not in t_remain:
#            t_remain.add(t)
#        t_id+=1
#                
#new_samples=[]
#new_samples_token=[]
#new_samples_rep_token=[]
#for ex_id,sample in enumerate(samples_token):
#    e1_id,p_id,e2_id=sample['triple'][0]
#    t=(e1_id,p_id,e2_id)
#    if t in t_remain:
#        new_samples.append(samples[ex_id])
#        new_samples_token.append(sample)
#        new_samples_rep_token.append(samples_rep[ex_id])
#        
#print({'sparsity':sparsity})
#print({'t_remain':len(t_remain)},{'T':T},{'rate:':len(t_remain)/T})
#print({'origlen sample':len(new_samples)},{'new_samples':len(samples)},{'rate:':len(new_samples)/len(samples)})
#train_orig_new=train_orig.split('.')[0]+'_spa'+str(sparsity)+'.json' # 'train_spa0.9.json'
#train_token_new=train_token.split('.')[0]+'_spa'+str(sparsity)+'.json' # 'train_rep_tokenized_spa0.9.json'
#train_rep_token_new=train_rep_token.split('.')[0]+'_spa'+str(sparsity)+'.json' # 'train_rep_tokenized_spa0.9.json'
#with open(train_orig_new,'w') as f:
#    json.dump(new_samples,f)
#with open(train_token_new,'w') as f:
#    json.dump(new_samples_token,f)
#with open(train_rep_token_new,'w') as f:
#    json.dump(new_samples_rep_token,f)
                
    