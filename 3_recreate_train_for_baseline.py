#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 10:38:14 2018

@author: xuweijia
"""

import json
import copy
train_file='/home/xuweijia/my_drqa/data/train_before_statis_new.json'
with open(train_file,'r') as f:
    samples1=json.load(f)
    #samples1=samples1[:100]

train_file='/home/xuweijia/my_drqa/data/train_before_statis_tokenized.json'
with open(train_file,'r') as f:
    samples2=json.load(f)
    #samples2=samples2[:100]
train_file='/home/xuweijia/my_drqa/data/train_before_statis_rep_tokenized.json'
with open(train_file,'r') as f:
    samples_rep=json.load(f)    

# new ['phrase_tokens'], ([Q_tokens]/['Q_tokens_rep_mention']),['good'],['e1_pos'],['e2_pos']
new_origi_samples=[]   # 185000/185232/185850  (drop all e1==e2/ drop can't find pos/ origi)
new_token_samples=[]
new_token_rep_samples=[]
n_no_good=0
n_false_drop=0  # 213
n_total_drop=0  # 405
train_set=set() # 67637
true_drop=set() # 125
for ex_id,sample in enumerate(samples2):
    print(ex_id)
    sample1=samples1[ex_id]
    sample_rep=samples_rep[ex_id]
    
    assert sample['triple']==sample_rep['triple']
    
    e1,e2=sample['triple'][1][0],sample['triple'][1][2]
    e1_id,e2_id=sample['triple'][0][0],sample['triple'][0][2]
    triple=tuple(sample['triple'][0])
    if e1_id==e2_id:
        continue
    
    e1_in_can=-1
    ans_in_can=-1

    for c_idx,c in enumerate(sample['raw_can']):
        Q_pos=sample['can2Q'][c_idx]
        if len([pos for pos in Q_pos if sample['Q_id'][pos]==e1_id])>0:
                e1_mention=c
                e1_in_can=c_idx
            # e1_mention=c     
        if len([pos for pos in Q_pos if sample['Q_id'][pos]==e2_id])>0:
                ans_mention=c
                ans_in_can=c_idx  
                
    if e1_in_can==ans_in_can:
        if e1_id in sample1['Q_tokens']:
            pos=sample1['Q_tokens'].index(e1_id)
            if sample1['phrase_tokens'][pos]!=e1_mention:
                e1_mention=sample1['phrase_tokens'][pos]
                e1_in_can=sample['raw_can'].index(e1_mention)
        if e2_id in sample1['Q_tokens']:
            pos=sample1['Q_tokens'].index(e2_id)
            if sample1['phrase_tokens'][pos]!=ans_mention:
                ans_mention=sample1['phrase_tokens'][pos]
                ans_in_can=sample['raw_can'].index(ans_mention)       
        
    if e1_in_can==ans_in_can:
        n_no_good+=1
        sample['same_entity']=True
    else:
        sample['same_entity']=False
        
    assert e1_in_can!=-1
    assert ans_in_can!=-1
    # assert ans_in_can!=e1_in_can
    # keep original token pos
    have_e1=False
    Q_doc=copy.deepcopy(sample['document'])
    for c_idx,c_spans in enumerate(sample['can_span']): # ''
        tag='other'
        if c_idx==ans_in_can:
            tag='ans'
        if c_idx==e1_in_can:
            tag='e1'   
        #print(c_idx,tag,ans_in_can)
        for i_span,span in enumerate(c_spans):     # include end
            init_tag=tag
            if init_tag=='e1' and i_span!=0 and sample['same_entity']:# for same entity have same can,just give one span to e1/ans
                tag='ans'   
            start,end=span 
            for i in range(start,end+1): 
                Q_doc[i]=("@_{}_{}".format(tag,c_idx),Q_doc[i]) 
                
    
    sample['Q_middle']=Q_doc
    sample1['Q_middle']=Q_doc
    sample_rep['Q_middle']=Q_doc
    
    # replace all single tokens
    e1_pos=-1
    e2_pos=-1
    all_Q_tokens=[]
    all_phrase_tokens=[]
    idx=0
    while(idx<len(Q_doc)):
        t=Q_doc[idx]
        if isinstance(t,str):
            all_Q_tokens.append(t)
            all_phrase_tokens.append(t)
            idx+=1
        else:
            tag=t[0]
            c_idx=int(tag.split('_')[-1])
            if tag.startswith('@_e1_'):
                c=sample['raw_can'][e1_in_can]
                c_id=e1_id
                e1_pos=len(all_Q_tokens)
            elif tag.startswith('@_ans_'):
                c=sample['raw_can'][ans_in_can]
                c_id=e2_id
                e2_pos=len(all_Q_tokens)
            else:
                c=sample['raw_can'][c_idx]
                c_id=sample['Q_id'][sample['can2Q'][c_idx][0]] # no disamguation
            all_Q_tokens.append(c_id)
            all_phrase_tokens.append(c)
            #print(0,tag,c,c_id,idx)
            idx+=1
            if idx<len(Q_doc):
                while Q_doc[idx][0]==tag:
                    #print(1,c,c_id,idx)
                    if len(Q_doc[idx])==1:
                            break
                    else:
                        idx+=1
                        if idx==len(Q_doc):
                            break
    # drop can't find pos ex
    if (e1_pos==-1 or e2_pos==-1):
        if triple in train_set:
            n_false_drop+=1
        else:
            n_total_drop+=1
            true_drop.add(triple)
        continue
            
    assert e1_pos!=-1
    assert e2_pos!=-1
    assert all_Q_tokens[e1_pos]==e1_id
    assert all_Q_tokens[e2_pos]==e2_id
    
    sample['all_Q_tokens']={'e1_pos':e1_pos,'e2_pos':e2_pos,'Q_tokens':all_Q_tokens,'phrase_tokens':all_phrase_tokens}
    sample1['all_Q_tokens']={'e1_pos':e1_pos,'e2_pos':e2_pos,'Q_tokens':all_Q_tokens,'phrase_tokens':all_phrase_tokens}
    sample_rep['all_Q_tokens']={'e1_pos':e1_pos,'e2_pos':e2_pos,'Q_tokens':all_Q_tokens,'phrase_tokens':all_phrase_tokens}
        
#     # just replace e1/ans with Q
    Q_tokens=[]
    phrase_tokens=[]    
    e1_pos=-1
    e2_pos=-1
    idx=0
    while(idx<len(Q_doc)):
        t=Q_doc[idx]
        if isinstance(t,str):
            Q_tokens.append(t)
            phrase_tokens.append(t)
            idx+=1
        elif '@_other' in t[0]:
            t=t[1]
            while not isinstance(t,str):
                t=t[-1]
            Q_tokens.append(t)
            phrase_tokens.append(t)
            idx+=1
        else:
            tag=t[0]
            c_idx=int(tag.split('_')[-1])
            if tag.startswith('@_e1_'):
                c=sample['raw_can'][e1_in_can]
                c_id=e1_id
                e1_pos=len(Q_tokens)
            elif tag.startswith('@_ans_'):
                c=sample['raw_can'][ans_in_can]
                c_id=e2_id
                e2_pos=len(Q_tokens)
            Q_tokens.append(c_id)
            phrase_tokens.append(c)
            #print(0,tag,c,c_id,idx)
            idx+=1
            if idx<len(Q_doc):
                while Q_doc[idx][0]==tag:
                    #print(1,c,c_id,idx)
                    if len(Q_doc[idx])==1:
                            break
                    else:
                        idx+=1
                        if idx==len(Q_doc):
                            break
    assert e1_pos!=-1
    assert e2_pos!=-1
    assert Q_tokens[e1_pos]==e1_id
    assert Q_tokens[e2_pos]==e2_id   
    sample['only_Q_tokens']={'e1_pos':e1_pos,'e2_pos':e2_pos,'Q_tokens':Q_tokens,'phrase_tokens':phrase_tokens}
    sample1['only_Q_tokens']={'e1_pos':e1_pos,'e2_pos':e2_pos,'Q_tokens':Q_tokens,'phrase_tokens':phrase_tokens}
    sample_rep['only_Q_tokens']={'e1_pos':e1_pos,'e2_pos':e2_pos,'Q_tokens':Q_tokens,'phrase_tokens':phrase_tokens}
    
    sample['E_info']={'e1_in_can':e1_in_can,'ans_in_can':ans_in_can,'e1_mention':e1_mention,'ans_mention':ans_mention}
    sample1['E_info']={'e1_in_can':e1_in_can,'ans_in_can':ans_in_can,'e1_mention':e1_mention,'ans_mention':ans_mention}
    sample_rep['E_info']={'e1_in_can':e1_in_can,'ans_in_can':ans_in_can,'e1_mention':e1_mention,'ans_mention':ans_mention}
    
    
    # sample_rep['only_Q_tokens']={'e1_pos':e1_pos,'e2_pos':e2_pos,'Q_tokens':Q_tokens,'phrase_tokens':phrase_tokens}
    
    train_set.add(triple)
    new_origi_samples.append(sample1)
    new_token_samples.append(sample)
    new_token_rep_samples.append(sample_rep)
    
import subprocess
final_train_dir='data/final_train/'
subprocess.call(['mkdir', '-p', final_train_dir])   
with open(final_train_dir+'train.json','w') as f:
    json.dump(new_origi_samples,f)
with open(final_train_dir+'train_tokenized.json','w') as f:
    json.dump(new_token_samples,f)
with open(final_train_dir+'train_rep_tokenized.json','w') as f:
    json.dump(new_token_rep_samples,f)    
    
    
    
    