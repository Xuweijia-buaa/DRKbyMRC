#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 22:16:36 2018

@author: xuweijia
"""
import argparse
import json
import time
from spacy_tokenizer import SpacyTokenizer
from multiprocessing import Pool
from multiprocessing.util import Finalize
from functools import partial
from collections import OrderedDict
import numpy as np
from utils import find_answer,find_span,tokenize2
# delete samples which have can couldn't match offsets
TOK = None
def init(tokenizer_class, options):
    global TOK
    TOK = tokenizer_class(**options)   # a SpacyTokenizer object
    Finalize(TOK, TOK.shutdown, exitpriority=100)

# return a dict, each property have text's all tokens
def tokenize(text):
    """Call the global process tokenizer on the input text."""
    global TOK
    tokens = TOK.tokenize(text)     # a Token object
    output = {
        'words': tokens.words(),     # 'Obama',...
        'offsets': tokens.offsets(), # (0,6),...
        'pos': tokens.pos(),         # 'NNR',...
        'lemma': tokens.lemmas(),    # 'obama',...
        'ner': tokens.entities(),    # 'O',...
        'untokenize':tokens.untokenize(),
        # 'entities':tokens.entity_groups(), #  [（"mao ze dong"，ORG),（"jing gang shan"，ORG）]  
    }
    return output

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')
parser = argparse.ArgumentParser()
parser.register('type', 'bool', str2bool)
parser.add_argument('--single_token', type='bool', default=True)     #   all split into single tokens  /  use NER represent doc
parser.add_argument('--workers', type=int, default=None)
parser.add_argument('--tokenizer', type=str, default='spacy')        #  tokenizer
parser.add_argument('--if_train_file', type='bool', default=True)    #   if is train file
parser.add_argument('--if_use_anchor', type='bool', default=True)    #   train: use valid_candis as candidates 
args = parser.parse_args()

t0 = time.time()

input_file='data/train_before_statis.json'
with open(input_file,'r') as f:
    samples = json.load(f)

import copy
from collections import Counter

new_samples=[]
for sample in samples:
# fix unvalid mention name
    flag_del=False
    M=[]
    for m in sample['candis_links']:
        if m.strip()!=m and m.strip() in sample['candis_links']:
            M.append(m)
            Q_m=sample['candis_links'][m]
            Q_m= [Q for Q in Q_m if Q_m not in sample['candis_links'][m.strip()]]
            sample['candis_links'][m.strip()].extend(Q_m)
            flag_del=True
        elif m.strip()!=m:
            Q_m=sample['candis_links'].pop(m)
            sample['candis_links'][m.strip()]=Q_m
        if m.strip('"')!=m and m.strip('"') in sample['candis_links']:
            M.append(m)
            Q_m=sample['candis_links'][m]
            Q_m= [Q for Q in Q_m if Q_m not in sample['candis_links'][m.strip('"')]]
            sample['candis_links'][m.strip('"')].extend(Q_m)
            flag_del=True            
    if flag_del:
        for m in M:
            sample['candis_links'].pop(m)
# del one Q's multi name            
    sample['candis_links_new']=copy.copy(sample['candis_links'])
    for k in sample['candis_links']:
        v=sample['candis_links'][k]
        Q_id=[e[0] for e in v]
        Q_name=[e[1] for e in v]
#        if len(set(Q_name))!=len(Q_name):
#            print('what???')
#            raise ValueError
#            break
        # multi, del one [Q_id,Q_name]
        if len(set(Q_id))!=len(Q_id):
            multi_Qid=[kk for kk ,vv in dict(Counter(Q_id)).items() if vv>1]
            drop_index=[]
            for e_id in multi_Qid:
                pos=np.where(np.array(Q_id)==e_id)[0]
                multi_name=list(np.array(Q_name)[pos])
                if sample['answear'] in multi_name:
                    # drop other name have same e_id
                    drop_index.extend([i for i in pos if Q_name[i]!=sample['answear']])
                else:
                    # shortest name
                    leng=[len(name) for name in multi_name]
                    keep_name=multi_name[leng.index(min(leng))]
                    drop_index.extend([i for i in pos if Q_name[i]!=keep_name])
            sample['candis_links_new'][k]= [m for m in v if v.index(m) not in drop_index]
    sample['candis_links']=sample['candis_links_new']
    sample['valid_cands']=list(sample['candis_links'].keys())
    
    sample['ans_mention']=sample['ans_mention'].strip()
    if sample['ans_mention'].strip('"') in sample['valid_cands']:
         sample['ans_mention']=sample['ans_mention'].strip('"')
    new_samples.append(sample) 
    
samples=new_samples

with open('data/Q_desp.json','r') as f:
    Q_desp=json.load(f)
    Q_desp=OrderedDict(Q_desp)
    
def need_token(samples):
#   if args.single_token:  # doc all split into single tokens
    if True:
        #  each token have offset,features
        output = {'qids': [], 'contexts': [], 'questions': [],'cands':[], 'surface_ans': [], 'ans': []}
        can_span=[]
        s=0
        for i,sample in enumerate(samples):
            output['qids'].append(i)
            output['questions'].append(sample['query'])  #  sample['e1']+ sample['query']+ sample['r_des'],  sample['query']+ sample['r_des'],  sample['query'], sample['r_des'],sample['e1']  
            output['contexts'].append(sample['doc'])
            output['surface_ans'].append(sample['ans_mention'])
            if sample['ans_mention'] not in sample['valid_cands']:
                print('what!!!')
                break
            can_span.append([s,s])
            for c in sample['valid_cands']:
               output['cands'].append(c)
               s+=1
            can_span[-1][1]=s
        output['cands_span']=can_span
        output['desp']=list(Q_desp.values())
    return output

data=need_token(samples)
make_pool = partial(Pool, args.workers, initializer=init)
tokenizer_class=SpacyTokenizer
# doc 
workers = make_pool(initargs=(tokenizer_class, {'annotators': {'lemma', 'pos', 'ner'}}))
c_tokens = workers.map(tokenize, data['contexts'])
workers.close()
workers.join()
# question
workers = make_pool(initargs=(tokenizer_class, {'annotators': {'lemma'}}))
q_tokens = workers.map(tokenize, data['questions'])
workers.close()
workers.join()
# cands
workers = make_pool(initargs=(tokenizer_class, {'annotators': {'lemma'}}))
cd_tokens = workers.map(tokenize, data['cands'])
workers.close()
workers.join()
# ans_mention
workers = make_pool(initargs=(tokenizer_class, {'annotators': {'lemma'}}))
a_tokens = workers.map(tokenize, data['surface_ans'])
workers.close()
workers.join()
# desp
workers = make_pool(initargs=(tokenizer_class, {'annotators': {'lemma'}}))
desp_tokens = workers.map(tokenize, data['desp'])
workers.close()
workers.join()

Q_desp_new=dict(zip(Q_desp.keys(),  [tokens['words'] for tokens in desp_tokens]))

new_list=[]
new_list_sample=[]
import copy
n_drop=0
n_drop_NER=0
drop=[]
drop_ner=[]
for idx in range(len(data['qids'])):
    print("n:{}".format(idx))
    sample=samples[idx]
    # ques
    question = q_tokens[idx]['words']
    qlemma = q_tokens[idx]['lemma']
    # doc
    document = c_tokens[idx]['words']
    offsets = c_tokens[idx]['offsets']
    lemma = c_tokens[idx]['lemma']
    POS = c_tokens[idx]['pos']
    ner = c_tokens[idx]['ner']
    
#  1 mention score: each cand start/end                    (each can's all pos start/end score *  softmax over mention) each span linked can    each can's linked E   (all E's desp)
#  2 mention score: each cand token contained words in doc (all doc words in can_word's score/    softmax over mention) each span linked can    each can's linked E   (all E's desp)
    
    # can: split can and it's raw can 
    can=[]                                   # [['New','York'],['New','York','City']]
    raw_can=[]                               # ['New York','New York City']
    ans_in_can=-1                            # ans mention pos among candidates
    start,end=data['cands_span'][idx]
    # ii: ii th cand      i: can's pos in all can
    for ii,poss in enumerate(range(start,end)):
        can.append(cd_tokens[poss]['words'])
        raw_can.append(sample['valid_cands'][ii])
        if sample['ans_mention']==raw_can[-1]:
            ans_in_can=ii
        assert sample['valid_cands'][ii]==cd_tokens[poss]['untokenize']
    assert ans_in_can!=-1    
    
    # mention score
    # each can's all span in doc   span[idx]: all pos of idx th cand of this sample
    span=[]                                    
    for c in raw_can:
        char_span=find_span(sample['doc'],c)       # each mention's all char_span [[0, 3], [7, 10]]
        span_c=[]                                  # each mention's all token span (start,end pos in doc)
        for start,end in char_span:
            found=find_answer(offsets, start,end)  # each mention's each char span offset:  (0,5)/(6,8),(9,10)/.../
            if found:
                span_c.append(found)
            else:
                continue
        if len(span_c)==0:
            flag_drop=True
            break
        else:
            flag_drop=False
        span.append(span_c)
        
    if flag_drop==True:
        n_drop+=1
        drop.append(idx)
        continue
    
    # each ex's Q_name, Q_desp and ans_in_Q     (every Q )   (in batch, summerize all Q desp,compute score betewwn encoded Q_desp and docs)  dict1.update(dict2) 
    Q_set=set()
    ans_in_Q=-1
    for c in raw_can:
        Q=sample['candis_links'][c]
        Q_set|=set(tuple([tuple(e) for e in Q]))
    Q_set=list(Q_set)
    Q_id=[e[0] for e in Q_set]
    Q_name=[e[1].split() for e in Q_set]
    Q_desp=[Q_desp_new[e[0]] for e in Q_set]
    ans_in_Q=Q_id.index(sample['triple'][2])
    assert ans_in_Q!=-1

    #  token replaced by mention (E name contain -,.)
    doc=copy.copy(sample['doc'])
    resort_can=sorted(raw_can, key=lambda x:len(x),reverse=True)     # raw_can
    for c in resort_can:
        doc=doc.replace(' '+c+' ',' '+'@'+'_'.join(c.split())+' ')
        doc=doc.replace(' '+c+',',' '+'@'+'_'.join(c.split())+',')
        doc=doc.replace(' '+c+'.',' '+'@'+'_'.join(c.split())+'.')
        if doc.startswith(c+' '):
            doc=doc.replace(c+' ','@'+'_'.join(c.split())+' ')
        if doc.startswith(c+','):
            doc=doc.replace(c+',','@'+'_'.join(c.split())+',')
        if '-' in doc or ':' in doc or ';' in doc:
            doc=doc.replace(' '+c+'-',' '+'@'+'_'.join(c.split())+'-')
            doc=doc.replace('('+c+'-','('+'@'+'_'.join(c.split())+'-')
            doc=doc.replace(' '+c+':',' '+'@'+'_'.join(c.split())+':')
            doc=doc.replace(' '+c+';',' '+'@'+'_'.join(c.split())+';')
        if '(' in doc or ')' in doc:
            doc=doc.replace('('+c+')','('+'@'+'_'.join(c.split())+')')
            doc=doc.replace(' '+c+')',' '+'@'+'_'.join(c.split())+')')
            doc=doc.replace('('+c+' ','('+'@'+'_'.join(c.split())+' ')
            doc=doc.replace('('+c+',','('+'@'+'_'.join(c.split())+',')
        if '"' in doc:
            doc=doc.replace('"'+c+'"','"'+'@'+'_'.join(c.split())+'"')
            doc=doc.replace('"'+c+',','"'+'@'+'_'.join(c.split())+',')
            doc=doc.replace(' '+c+'"',' '+'@'+'_'.join(c.split())+'"')
            doc=doc.replace('"'+c+' ','"'+'@'+'_'.join(c.split())+' ')
            # doc=doc.replace("'"+c+"'","'"+'@'+'_'.join(c.split())+"'")
        if "'s" in doc :
            doc=doc.replace(' '+c+"'s" ,' '+'@'+'_'.join(c.split())+"'s" )
            doc=doc.replace('"'+c+"'s" ,'"'+'@'+'_'.join(c.split())+"'s" )
    doc_tokens=tokenize2(doc)                                        # split doc tokens
    document_rep = doc_tokens['words']
    offsets_rep = doc_tokens['offsets']
    lemma_rep = doc_tokens['lemma']
    pos_rep = doc_tokens['pos']
    ner_rep = doc_tokens['ner']        
        
    # token_replaced_by_NER can
    ans_in_can_rep=-1
    pos_c=[]                                                        # can all pos in doc
    for c in resort_can:
        if sample['ans_mention']==c:
            ans_in_can_rep=resort_can.index(sample['ans_mention'])
        cc= '@'+'_'.join(c.split())
        pos_cc=[i for i in range(len(document_rep)) if document_rep[i]==cc]
        if len(pos_cc)==0:
            flag_drop=True
            break
        else:
            flag_drop=False
        pos_c.append(pos_cc)
        
    if flag_drop==True:
        n_drop_NER+=1
        drop_ner.append(idx)
        continue
    assert ans_in_can_rep!=-1 

    if args.single_token:
        # mention 2 Q
        c2Q=[]
        for c in raw_can:
            Q=sample['candis_links'][c]
            Q_list=list(set(tuple([e[0] for e in Q])))   # this can's all Q
            c2Q_pos=[Q_id.index(e_id) for e_id in Q_list if e_id in Q_id]
            c2Q.append(c2Q_pos)
            
        sample_new= {
            'id': idx,                      # this ques id
            'question': question,           # all ques tokens
            'qlemma': qlemma,               # all ques tokens's lemma
            'document': document,           # all doc  tokens
            'offsets': offsets,             # all tokens's span in doc
            'lemma': lemma,                 # all doc  tokens's lemma
            'pos': POS,                     # all doc  tokens's pos
            'ner': ner,                     # all doc  tokens's ner
            'raw_can': raw_can,             # n_can: all valid cands
            'can':can,                      # n_can,max_can_tokens:  all split valid cands, mask doc, to compute each mention's score              (method 2, like GA)
            'can_span':span,                # n_can,max_can_show,2:   all can's all token's span, used to compute each mention score by start * end (method 1) 
            'Q_id':Q_id,                    # each Q's id
            'Q_name':Q_name,                # each Q's name (splited)
            'Q_desp':Q_desp,                # each Q's desp (splited)
            'can2Q':c2Q,                    # each can's correspond Q's pos  [2,3,4],[2,4],[1,2,3]
            'ans_in_can':ans_in_can,        # ans mention pos in can .  can get ans mention tokens span by this pos.
            'ans_in_Q':ans_in_Q,            # ans pos in Q_set .        can get ans information by this pos.
            'ans_name':sample['answear'],   # ans_name
            'ans_id':sample['triple'][2],   # ans_id
            'triple':[sample['triple'],[sample['e1'],sample['query'],sample['answear']]]
        }
    else:
        # mention 2 Q
        c2Q=[]
        for c in resort_can:
            Q=sample['candis_links'][c]
            Q_list=list(set(tuple([e[0] for e in Q])))
            c2Q_pos=[Q_id.index(e_id) for e_id in Q_list if e_id in Q_id]
            c2Q.append(c2Q_pos)
        
        sample_new= {
            'id': idx,                      # this ques id
            'question': question,           # all ques tokens
            'qlemma': qlemma,               # all ques tokens's lemma
            'document': document_rep,       # all doc  tokens
            'offsets': offsets_rep,             # all tokens's span in doc
            'lemma': lemma_rep,                 # all doc  tokens's lemma
            'pos': pos_rep,                     # all doc  tokens's pos
            'ner': ner_rep,                     # all doc  tokens's ner
            'raw_can': resort_can,           # n_can: all valid cands
            'can_pos':pos_c,                # n_can,max_can_show,2:   all can's all pos in doc, used to compute each mention score by sum them (method 3) 
            'Q_id':Q_id,                    # each Q's id
            'Q_name':Q_name,                # each Q's name (splited)
            'Q_desp':Q_desp,                # each Q's desp (splited)
            'can2Q':c2Q,                    # each can's correspond Q's pos  [2,3,4],[2,4],[1,2,3]
            'ans_in_can':ans_in_can_rep,        # ans mention pos in can .  can get ans mention tokens span by this pos.
            'ans_in_Q':ans_in_Q,            # ans pos in Q_set .        can get ans information by this pos.
            'ans_name':sample['answear'],   # ans_name
            'ans_id':sample['triple'][2],   # ans_id
            'triple':[sample['triple'],[sample['e1'],sample['query'],sample['answear']]]
        }        
    new_list.append(sample_new)
    new_list_sample.append(sample)


print('drop rate:{}'.format((n_drop+n_drop_NER)/len(samples)))

# input_file='data/train_before_statis.json'
if args.single_token:
    # use to train
    output_file=input_file.split('.')[0]+'_tokenized.json'
    final_file='data/train_before_statis_new.json'
else:
    output_file=input_file.split('.')[0]+'_rep_tokenized.json'
    final_file='data/train_before_statis_rep_new.json'
    
with open(output_file,'w') as f:
    json.dump(new_list,f)
    
# final train
with open(final_file,'w') as f:
    json.dump(new_list_sample,f)


# n_old :219907 n_new  218266   n_drop 1642    drop rate:0.0074667588264183205
# n_old :219907 n_new  185850   n_drop2 32416  drop rate:0.1548
