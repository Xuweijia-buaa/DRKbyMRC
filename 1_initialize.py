#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 22:55:40 2018

@author: xuweijia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:17:42 2018

@author: xuweijia
"""
# delete mention either Q1/Q2 couldn't be replaced

import json
# add  sample['ans_mention']=c            # ans's correspond meniton
#      sample['c_pos']=i                  # ans pos in this mention's candidates

train_file='/media/xuweijia/06CE5DF1CE5DD98F/useful_old_data/new_NER_data/e_30/train_20w_list.json'
with open(train_file,'r') as f:
    samples=json.load(f)
# add  ['Q_tokens_rep_mention'],['good'],['e1_mention'],['ans_mention'],['e1_pos'],['e2_pos']
# del 
import copy
import nltk
#def find_and_replace_e1_e2_mention(samples):
n_m_e1=0
n_m_e2=0
good=0
new_list=[]
wrong_list=[]
for n,sample in enumerate(samples):
    e1_id,p_id,e2_id=sample['triple']
    if e1_id==e2_id:
        continue
    count_e1=0
    count_e2=0
    # 1 : find mention
    # all in Q tokens
    sample['good']=0
    if e1_id in sample['Q_tokens'] and e2_id in sample['Q_tokens']:
         good+=1
         count_e1=1
         count_e2=1
         e1_pos=sample['Q_tokens'].index(e1_id)
         e2_pos=sample['Q_tokens'].index(e2_id)
         sample['e1_mention']=sample['phrase_tokens'][e1_pos]       # e1's correspond meniton        
         sample['ans_mention']=sample['phrase_tokens'][e2_pos]      # ans's correspond meniton
         sample['good']=1
         sample['Q_tokens_rep_mention']=copy.deepcopy(sample['phrase_tokens'])
         sample['Q_tokens_rep_mention'][e1_pos] = e1_id
         sample['Q_tokens_rep_mention'][e2_pos] = e2_id
         
    # 2 mention in Q2anchor/ candi, and in phrase_tokens
    if sample['good']==0:
        for e_id, mention in sample['Q2anchor'].items():
           if e_id==e1_id:
                count_e1=1
                sample['e1_mention']=mention
           if e_id==e2_id:
                count_e2=1
                sample['ans_mention']=mention
        if sample.get('e1_mention')==None:
            for i,c in enumerate(sample['valid_cands']):  # sample['candis']
                Q_ids=sample['candis_links'][c]           # [['Q180322', 'Aragorn']]
                Q_ids=[e[0] for e in Q_ids]               # [Q1,Q2,...]            
                if e1_id in Q_ids:
                    count_e1+=1
                    sample['e1_mention']=c                # e1's correspond meniton            
        if sample.get('ans_mention')==None:
            for i,c in enumerate(sample['valid_cands']):  # sample['candis']
                Q_ids=sample['candis_links'][c]           # [['Q180322', 'Aragorn']]
                Q_ids=[e[0] for e in Q_ids]               # [Q1,Q2,...]                    
                if e2_id in Q_ids:
                    count_e2+=1
                    sample['ans_mention']=c               # ans's correspond meniton
    # 3 test multiple mention
        if count_e1==0 or count_e2==0:
            print('what 1')
            print(sample['candis_links'])
            print(e1_id,e2_id)
            break
        if count_e1>1:
            n_m_e1+=1
        if count_e2>1:
            n_m_e2+=1
        if sample['e1_mention'] in sample['phrase_tokens'] and sample['ans_mention'] in sample['phrase_tokens']:  # just replace mention with Q
            good+=1
            sample['good']=1
            e1_pos=sample['phrase_tokens'].index(sample['e1_mention'])
            e2_pos=sample['phrase_tokens'].index(sample['ans_mention'])
            sample['Q_tokens_rep_mention']=copy.deepcopy(sample['phrase_tokens'])
            sample['Q_tokens_rep_mention'][e1_pos] = e1_id
            sample['Q_tokens_rep_mention'][e2_pos] = e2_id              
        else:
            doc=copy.deepcopy(sample['phrase_tokens'])
            if sample['e1_mention'] in doc:
                e1_pos=doc.index(sample['e1_mention'])
                doc[e1_pos]=e1_id
            elif sample['ans_mention'] in doc:
                e2_pos=doc.index(sample['ans_mention'])
                doc[e2_pos]=e2_id
            doc=' '.join(doc)
            doc=doc.replace(sample['e1_mention'],e1_id)
            doc=doc.replace(sample['ans_mention'],e2_id)
            doc=doc.replace('-',' ')
            sample['Q_tokens_rep_mention']=nltk.word_tokenize(doc) # doc.split()
            if e1_id in sample['Q_tokens_rep_mention'] and e2_id in sample['Q_tokens_rep_mention']:
                e1_pos=sample['Q_tokens_rep_mention'].index(e1_id)
                e2_pos=sample['Q_tokens_rep_mention'].index(e2_id)
    
    # just keep both e1,e2 in sen
    if e1_id in sample['Q_tokens_rep_mention'] and e2_id in sample['Q_tokens_rep_mention']:
        sample['e1_pos']=e1_pos
        sample['e2_pos']=e2_pos
        new_list.append(sample)


good_rate=good/len(samples)
keep_rate=len(new_list)/len(samples)
e1_mul=n_m_e1/len(samples)
e2_mul=n_m_e2/len(samples)  


print("good rate:",good_rate)
print("keep rate:",keep_rate)
print("how many samples man have multiple answear:", e1_mul)
print("how many samples man have multiple answear:", e2_mul)

with open('data/train_before_statis.json','w') as f:
    json.dump(new_list,f)
    
#good rate: 0.8365670038453276
#keep rate: 0.9642169694259194
#how many samples man have multiple answear: 0.0013110067567271308
#how many samples man have multiple answear: 0.0017143934511047096