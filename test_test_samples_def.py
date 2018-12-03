#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 19:33:00 2018

@author: xuweijia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:17:42 2018

@author: xuweijia
"""
# just test final samples (have ans rate), ans_mention linked E number, etc...
import json
import copy
from collections import Counter
import numpy as np
import argparse

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')
parse = argparse.ArgumentParser(description='call retriver')
parse.register('type', 'bool', str2bool)
parse.add_argument('--test_file', type=str, help='.../test_file,already contain triples')
parse.add_argument('--valid_sample', type='bool',default=True, help='.../test_file,already contain triples')
parse.add_argument('--mode',type=str2bool,default='dev',help='dev/test')
args = parse.parse_args()
if args.mode=='dev':
    mode='dev'
else:
    mode='test'

with open(args.test_file,'r') as f:
    samples=json.load(f)

n_anchor_notin=0                       # anchor not all in original
n_valid_notin=0                        # valid not all in original
n_c=0                                  # all mention
n_c_entity=0
n_a=0                                  # ans mention
n_a_entity=0

n_mul_m=0
new_list=[]

n_ans=0                               # how many samples contain more than 1 ans
n_ans_claim=0                         # average ans each sample contain
n_ans_real=0                          # how many samples contain more than 1 ans real
n_ans_claim_real=0                    # average ans each sample contain real

n_ans_each=0                          # how many samples contain real more than 1 ans
n_doc=0                               # all docs in all samples
n_no_ans=0                            # how many samples no ans

n_can=0
new_list=[]

count_p=dict()                       # each p's answear
count_p_count=dict()                 # each p's answear's linked entity number

# fix samples,include sample['candis_links'],sample['valid_cands']    wrong mention/ wrong_Q_name
new_samples=[]
for sample in samples:
    sample['candis_links_new']=copy.copy(sample['candis_links'])
    for i in range(len(sample['docs'])):
        # 1 fix unvalid mention name
        flag_del=False
        M=[]
        for m in sample['candis_links'][i]:
            if m.strip()!=m and m.strip() in sample['candis_links'][i]:
                M.append(m)
                Q_m=sample['candis_links'][i][m]
                Q_m= [Q for Q in Q_m if Q_m not in sample['candis_links'][i][m.strip()]]
                sample['candis_links'][i][m.strip()].extend(Q_m)
                flag_del=True
            elif m.strip()!=m:
                Q_m=sample['candis_links'][i].pop(m)
                sample['candis_links'][i][m.strip()]=Q_m
            if m.strip('"')!=m and m.strip('"') in sample['candis_links'][i]:
                M.append(m)
                Q_m=sample['candis_links'][i][m]
                Q_m= [Q for Q in Q_m if Q_m not in sample['candis_links'][i][m.strip('"')]]
                sample['candis_links'][i][m.strip('"')].extend(Q_m)
                flag_del=True 
        if flag_del:
            for m in M:
                sample['candis_links'][i].pop(m)
        # 2 del one Q's multi name            
        for k in sample['candis_links'][i]:
            v=sample['candis_links'][i][k]
            Q_id=[e[0] for e in v]
            Q_name=[e[1] for e in v]
            if len(set(Q_id))!=len(Q_id):
                multi_Qid=[kk for kk ,vv in dict(Counter(Q_id)).items() if vv>1]
                drop_index=[]
                for e_id in multi_Qid:
                    pos=np.where(np.array(Q_id)==e_id)[0]
                    multi_name=list(np.array(Q_name)[pos])
                    if any([a for a in sample['ans_list'] if a in multi_name]):
                        # drop other name have same e_id
                        drop_index.extend([i for i in pos if Q_name[i] not in sample['ans_list']])
                    else:
                        # shortest name
                        leng=[len(name) for name in multi_name]
                        keep_name=multi_name[leng.index(min(leng))]
                        drop_index.extend([i for i in pos if Q_name[i]!=keep_name])
                sample['candis_links_new'][i][k]= [m for m in v if v.index(m) not in drop_index]
        sample['candis_links'][i]=sample['candis_links_new'][i]
        sample['valid_cands'][i]=list(sample['candis_links'][i].keys())
    new_samples.append(sample)  
samples=new_samples

for n,sample in enumerate(samples):
    
    valid_c=sample['valid_cands'] if args.valid_sample else sample['candis']
    
    ans=sample['ans_list']
    p_id=sample['p_id']
    p_label=sample['query']
    e2_ids=sample['ans_id']
    if len(e2_ids)<=1:
        sample['multi_ans']=False
        n_ans_claim+=1
    else:
        n_ans+=1 
        n_ans_claim+=len(e2_ids)
        sample['multi_ans']=True    
        
    n_doc+=len(sample['docs'])  
    
    can_set=set()                                      # all can for this sample  
    count=0
    sample['ans_mention']=[]
    ans_set=set()
    sample['ans_mention']=[ [] for m in range(len(sample['docs']))]
    for i,doc in enumerate(sample['docs']):
        true_Q=list(sample['Q2anchor'][i].values())   # all entity real in this doc
        n_not_in=False                                # if all anchor found in this doc
        can_set|=set(sample['valid_cands'][i])
        if len(true_Q)!=0:
            for c in true_Q:
                if c not in sample['orig_cands'][i]:
                    n_not_in=True
                    break
        if n_not_in:
            n_anchor_notin+=1                        #

        # test how many valid not in NER         
        n_not_in=False
        for c in sample['valid_cands'][i]:
            if c not in sample['orig_cands'][i]:
                n_not_in=True
                break
        if n_not_in:
            n_valid_notin+=1  
            
        # compute avrage entity each mention linked
        for c in valid_c[i]:
            if len(sample['candis_links'][i][c])!=1:
                n_c+=1
                n_c_entity+=len(sample['candis_links'][i][c])

        # find surface mention for each ans:
        for ans_i,e2_id in enumerate(e2_ids):         # each ans  
            for e_id, mention in sample['Q2anchor'][i].items():
                if mention.strip('"') in sample['candis_links'][i]:
                    mention=mention.strip('"')
                if e_id==e2_id:
                    count+=1
                    sample['ans_mention'][i].append([mention,e2_id])         
                    ans_set.add(e2_id)
                    n_a+=1
                    n_a_entity+=len(sample['candis_links'][i][mention])
                    if count_p.get(p_id)!=None:
                       count_p[p_id]+=1
                       count_p_count[p_id]+=len(sample['candis_links'][i][mention])
                    else:
                       count_p[p_id]=1
                       count_p_count[p_id]=len(sample['candis_links'][i][mention])     
                      
            if len(sample['ans_mention'][i]) < ans_i+1:
                for c in valid_c[i]:   # sample['candis']
                    Q_ids=sample['candis_links'][i][c]
                    Q_ids=[e[0] for e in Q_ids]
                    if e2_id in Q_ids:
                        count+=1
                        n_a+=1
                        n_a_entity+=len(sample['candis_links'][i][c])
                        sample['ans_mention'][i].append([c,e2_id])   # ans_pos in all ans, ans's correspond meniton, ans pos in this mention's candidates
                        ans_set.add(e2_id)
                        if count_p.get(p_id)!=None:
                           count_p[p_id]+=1
                           count_p_count[p_id]+=len(sample['candis_links'][i][c])
                        else:
                           count_p[p_id]=1
                           count_p_count[p_id]=len(sample['candis_links'][i][c])         
                
    if len(ans_set)<=1:
        sample['multi_ans_real']=False
        n_ans_claim_real+=1
    else:
        n_ans_real+=1 
        n_ans_claim_real+=len(ans_set)     # average ans (real)
        sample['multi_ans_real']=True     
               
    if count==0:
        n_no_ans+=1
        sample['no_ans']=True
    else:
        sample['no_ans']=False
        
    n_can+=len(can_set)
    #print("n:{}".format(n))
    new_list.append(sample)

# count different p's ans menton avr linked entity number
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
        
    
avrage_entity=n_c_entity/n_c            # avrage entity each mention linked
avrage_ans_entity=n_a_entity/n_a        # avrage entity each ans mention linked
avrage_can=n_can/len(samples)           # average can each sample have
anchor_fil_rate=n_anchor_notin/n_doc    # how many ahchor can't be NER
valid_fil_rate=n_valid_notin/n_doc      # how many valid candidas can't be NER
no_ans_rate=n_no_ans/len(samples) 

mul_ans_rate=n_ans/len(samples)         # how many samples contain more than 1 ans
avrage_ans=n_ans_claim/len(samples)     # average ans each sample contain real

mul_ans_rate_real=n_ans_real/len(samples)      # how many samples contain more than 1 ans real
avrage_ans_real=n_ans_claim_real/len(samples)  # average ans each sample contain real



print(mode,'valid cands:', args.valid_sample)
print("avrage entity each mention linked:" ,avrage_entity)
print("avrage entity each ans mention linked:" ,avrage_ans_entity)
print("average can each sample have:" ,avrage_can)
print("have_ans_rate:" ,1-no_ans_rate)
print("how many ahchor can't be NER:", anchor_fil_rate)
print("how many valid candidas can't be NER:",valid_fil_rate)
print("how many samples  have multiple answear:", mul_ans_rate)
print("how many samples  have multiple answear real:", mul_ans_rate_real)
print("average ans each sample contain:", avrage_ans)
print("average ans each sample contain real:", avrage_ans_real)

#with open('{}_valid_{}.json'.format(mode,valid_sample),'w') as f:
#    json.dump(new_list,f)

with open('new_data/dict_p_{}_{}.json'.format(args.mode,args.valid_sample),'w') as f:
    json.dump(new_dict,f)
    

