#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 23:04:57 2018

@author: xuweijia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 19:53:54 2018

@author: xuweijia
"""
# find each sample's each doc's  cand mentions, Q_id, Q_name, Q_desp, to form batch  
import json
import argparse
from multiprocessing import Pool
from multiprocessing.util import Finalize
from functools import partial
from collections import OrderedDict
from spacy_tokenizer import SpacyTokenizer
import copy
from utils import find_answer,find_span,tokenize2
PROCESS_TOK = None
def init(tokenizer_class, tokenizer_opts):
    global PROCESS_TOK
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
# return a Token object
def tokenize_text(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)

#test=True
#contain=True
#if test:
#    mode='test'
#else:
#    mode='dev'
def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')
parser = argparse.ArgumentParser()
parser.register('type', 'bool', str2bool)

parser.add_argument('--mode', type=str,default='dev' ,help='dev/test/no_e')
parser.add_argument('--contain', type='bool',default=True ,help='contain/n_e')
parser.add_argument('--single_token', type='bool', default=False,help='all split into single tokens  /  use NER represent doc')
parser.add_argument('--input_file', type=str)
parser.add_argument('--valid_can', type='bool', default=True,help='use valid_candis as candidates or not ')
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--tokenizer', type=str, default='spacy')        #  tokenizer
parser.add_argument('--gap', type=float, default=1.5,help='which p should become multi_test')
parser.add_argument('--mode',type=str,default='dev',help='dev/test')
args = parser.parse_args()

can_name= 'valid_cands' if args.valid_can else 'candis'

new_folder='data/final_test_50/'
# no_e_ndoc50_raw.json
# dev_contain_e_ndoc50_raw.json
args.input_file=new_folder+'{}_contain_e_ndoc50_raw.json'.format(args.mode) if args.contain else new_folder+'no_e_ndoc50_raw.json'
with open(args.input_file,'r') as f:
    samples=json.load(f) 
    
with open('data/Q_desp_doc50.json','r') as f:
    Q_desp=json.load(f)
    Q_desp=OrderedDict(Q_desp)

#P_dict_file='data/dict_p.json'
#P_dict_file='data/dict_p_{}_{}.json'.format(mode,args.valid_can)
#with open(P_dict_file,'r') as f:
#    P_dict=json.load(f)
#    
# tokenize doc,query,Q_desp
def need_token(samples):
    if args.single_token:  # doc all split into single tokens
        #  each token have offset,features
        output = {'qids': [],  'questions': [],'contexts': [],'cands':[]}  # ans_mention in cands
        doc_span=[]
        can_span=[]
        can_span_big=[]   
        p_d=0
        p_c=0
        for i,sample in enumerate(samples):
            output['qids'].append(i)
            output['questions'].append(sample['query'])  #  sample['e1']+ sample['query']+ sample['r_des'],  sample['query']+ sample['r_des'],  sample['query'], sample['r_des'],sample['e1']  

            # doc span [0,20],[20,40]   range(start,end)
            doc_span.append([p_d,p_d])
            for d in sample['docs']:
                output['contexts'].append(d)
                p_d+=1
            doc_span[-1][1]=p_d           

            # can mention span
            can_span_q=[]
            for i in range(len(sample['docs'])):
                can_span.append([p_c,p_c])
                span=[p_c,p_c]
                for c in sample[can_name][i]:
                    output['cands'].append(c)
                    p_c+=1
                span[-1]=p_c
                can_span_q.append(span)
                can_span[-1][1]=p_c               #  n_doc   can_span[s_id]:    [20-25],                     each doc's all can
            can_span_big.append(can_span_q)       #  n_query can_span_big[q_id]:[[0,20],[20,25],[25,30]]     q_id's each doc's all can 
        output['docs_span']=doc_span  
        output['cands_span']=can_span
        output['cands_span_big']=can_span_big
        output['desp']=list(Q_desp.values())
    else:
        output = {'qids': [],  'questions': [],'contexts': [],'cands':[]}  # ans_mention in cands
        doc_span=[]
        can_span=[]
        can_span_big=[]   
        p_d=0
        p_c=0
        for i,sample in enumerate(samples):
            output['qids'].append(i)
            output['questions'].append(sample['query'])  #  sample['e1']+ sample['query']+ sample['r_des'],  sample['query']+ sample['r_des'],  sample['query'], sample['r_des'],sample['e1']  
        output['desp']=list(Q_desp.values())       
    return output

data=need_token(samples)
make_pool = partial(Pool, args.workers, initializer=init)
tokenizer_class=SpacyTokenizer


# question
workers = make_pool(initargs=(tokenizer_class, {'annotators': {'lemma'}}))
q_tokens = workers.map(tokenize_text, data['questions'])
workers.close()
workers.join()
# desp
workers = make_pool(initargs=(tokenizer_class, {'annotators': {'lemma'}}))
desp_tokens = workers.map(tokenize_text, data['desp'])
workers.close()
workers.join()

if args.single_token: 
    # doc 
    workers = make_pool(initargs=(tokenizer_class, {'annotators': {'lemma', 'pos', 'ner'}}))
    d_tokens = workers.map(tokenize_text, data['contexts'])
    workers.close()
    workers.join()
    # cands
    workers = make_pool(initargs=(tokenizer_class, {'annotators': {'lemma'}}))
    c_tokens = workers.map(tokenize_text, data['cands'])
    workers.close()
    workers.join()

# Q 2 Q_desp tokens
Q_desp_new=dict(zip(Q_desp.keys(),  [tokens.words() for tokens in desp_tokens]))
new_list=[]
new_list_multi=[]     # only multi link entites
new_list_single=[]
new_samples=[]
n_in=0
for qid in data['qids']:
    #print("n:{}".format(qid))
    sample=samples[qid]
    p_label=sample['query']
    e2_ids=sample['ans_id']
    # ques
    question = q_tokens[qid].words()
    qlemma =   q_tokens[qid].lemmas()
    
    if args.single_token:
        # doc
        document=[]
        offsets = []
        lemma = []
        pos = []
        ner = []
        d_start,d_end=data['docs_span'][qid]
        for i, sid in enumerate(range(d_start,d_end)):
            document.append(d_tokens[sid].words())
            offsets.append(d_tokens[sid].offsets())
            lemma.append(d_tokens[sid].lemmas())
            pos.append(d_tokens[sid].pos())
            ner.append(d_tokens[sid].entities())
        
        #  1 mention score: each cand start/end                    (each can's all pos start/end score *  softmax over mention) each span linked can    each can's linked E   (all E's desp)
        #  2 mention score: each cand token contained words in doc (all doc words in can_word's score/    softmax over mention) each span linked can    each can's linked E   (all E's desp)
        
        # can  (all docs)       no void can/span/ 
        can=[]                                       # each doc    [['New','York'],['New','York','City']]
        raw_can=[]                                   # each doc    ['New York','New York City']
        span=[]                                      # each doc    all can's all token spans
                                                          #                                 d1      d2   d3
        cands_span_big_q=data['cands_span_big'][qid]      # all this query's can's pos   [[1,3],[4,5], [6,18]]
        for i in range(len(sample['docs'])):
            doc_can=[]
            raw_doc_can=[]
            cs=sample[can_name][i]
            start,end=cands_span_big_q[i]            #  i'th doc all can's start/end in c_tokens  [c1,c2,c3]    [1,3]
            # ii: ii th cand in raw_can              #                                            c1         c2            c3
            span_doc=[]                              # one doc's 3 cands token spans : [   [[3,6],[8,10]],   [],  [[4,5],[6,7],[16,17]]   ]
            # ii:    c's pos in valid_can
            # c_pos: c's pos in c_tokens
            for ii,c_pos in enumerate(range(start,end)):
                c=cs[ii]
                assert c==c_tokens[c_pos].untokenize()
                span_c=[]                                  # this c's  all token spans, may be []
                char_span=find_span(sample['docs'][i],c)   # this c's  all char spans
                for start,end in char_span:
                    found=find_answer(offsets[i], start,end)  # each mention's token span
                    if found:
                        span_c.append(found)
                if len(span_c)==0:
                    continue
                doc_can.append(c_tokens[c_pos].words())      # doc_can: all  candidates:[['new','york'],['new','york','city']
                raw_doc_can.append(sample[can_name][i][ii])  # ['new york','new york city']
                span_doc.append(span_c)
#            if len(span_doc)==0:
#                continue
            can.append(doc_can)
            raw_can.append(raw_doc_can)
            span.append(span_doc)
    
        leave_pos=[i for i in range(len(sample['docs'])) if raw_can[i]!=[]]
        
        document=[document[i] for i in leave_pos]
        offsets=[offsets[i] for i in leave_pos]
        lemma=[lemma[i] for i in leave_pos]
        pos=[pos[i] for i in leave_pos]
        ner=[ner[i] for i in leave_pos]
        can=[can[i] for i in leave_pos]
        raw_can=[raw_can[i] for i in leave_pos]
        span=[span[i] for i in leave_pos]
        
        sample['docs']=[sample['docs'][i] for i in leave_pos]
        sample['candis_links']=[sample['candis_links'][i] for i in leave_pos]
        sample['valid_cands']=[sample['valid_cands'][i] for i in leave_pos]
        sample['candis']=[sample['candis'][i] for i in leave_pos]                                
        sample['phrase_tokens']=[sample['phrase_tokens'][i] for i in leave_pos]    
        sample['tokens']=[sample['tokens'][i] for i in leave_pos]
        sample['Q_tokens']=[sample['Q_tokens'][i] for i in leave_pos]
        sample['Q2anchor']=[sample['Q2anchor'][i] for i in leave_pos]
        sample['Q2title']=[sample['Q2title'][i] for i in leave_pos]
        
        assert [] not in raw_can
        #  each doc's Q_name, Q_desp and ans_in_Q     (every Q )   (in batch, summerize all Q desp,compute score betewwn encoded Q_desp and docs)  dict1.update(dict2)
        Q_id=[]
        Q_name=[]
        Q_desp=[]
        c2Q=[]          # c2Q[i][c_id_in_doc]: doc i's  c's all Q pos in Q_set
        ans_in_Q=[]     # [i]:  ans pos in doc i's Q_set
        ans_in_can=[]   # [i]:  ans mention pos in doc i's can
        ans_exist=False
        for i in range(len(sample['docs'])):
            Q_set=set()
            for c in raw_can[i]:
                Q=sample['candis_links'][i][c]
                Q_set|=set(tuple([tuple(e) for e in Q]))
            Q_set=list(Q_set)
            Q_id.append([e[0] for e in Q_set])
            Q_name.append([e[1].split() for e in Q_set])
            Q_desp.append([Q_desp_new[e[0]] if e[0] in Q_desp_new else Q_name[-1][idx] for idx,e in enumerate(Q_set)])
            if len(e2_ids)==1:
                if e2_ids[0] in Q_id[-1]:
                    ans_in_Q.append(Q_id[-1].index(e2_ids[0]))
                    ans_in_can.append(raw_can[i].index(c))    # ans_pos in doc i'th cand
                    ans_exist=True
                else:
                    ans_in_Q.append(-1)
                    ans_in_can.append(-1)
            else:
                if any ([e2_id for e2_id in e2_ids if e2_id in Q_id[-1]]):
                    ans_exist=True
                    correct_Q=[e2_id for e2_id in e2_ids if e2_id in Q_id[-1]]
                    ans_in_Q.append(Q_id[-1].index(correct_Q[0]))
                    ans_in_can.append(raw_can[i].index(c)) 
                else:
                    ans_in_Q.append(-1)
                    ans_in_can.append(-1)
            # mention 2 Q    each can's Q_pos  in Q_id/Q_name/Q_desp 
            c2Q_doc=[]
            for c in raw_can[i]:
                Q=sample['candis_links'][i][c]
                Q_list=list(set(tuple([e[0] for e in Q])))       # mention c's all Q
                c2Q_pos=[Q_id[-1].index(e_id) for e_id in Q_list]    # c's all Q's pos in doc Q 
                c2Q_doc.append(c2Q_pos)
            c2Q.append(c2Q_doc)
        
        # make process_dataset a generator（生成器) ,each time return a sample
        sample_new= {
            'ans_exist':ans_exist,
            'id': qid,                      # this ques id
            'question': question,           # all ques tokens
            'qlemma': qlemma,               # all ques tokens's lemma
            # all doc's
            'document': document,           # all doc  tokens
            'offsets': offsets,             # all tokens's span in doc
            'lemma': lemma,                 # all doc  tokens's lemma
            'pos': pos,                     # all doc  tokens's pos
            'ner': ner,                     # all doc  tokens's ner
            'raw_can': raw_can,             # n_can: all valid cands
            'can':can,                      # n_can,max_can_tokens:  all split valid cands, mask doc, to compute each mention's score              (method 2, like GA)
            'can_span':span,                # n_can,max_can_show,2:   all can's all token's span, used to compute each mention score by start * end (method 1) 
            'Q_id':Q_id,                    # each Q's id
            'Q_name':Q_name,                # each Q's name (splited)
            'Q_desp':Q_desp,                # each Q's desp (splited)
            'can2Q':c2Q,                    # each can's correspond Q's pos in Q_id/Q_name/Q_desp  [2,3,4],[2,4],[1,2,3]
            'ans_in_can':ans_in_can,        # ans mention pos in can .  can get ans mention tokens span by this pos.
            'ans_in_Q':ans_in_Q,            # ans pos in Q_set .        can get ans information by this pos.  0/pos
            'ans_name':sample['ans_list'],  # ans_name
            'ans_id':sample['ans_id'],      # ans_id
            'triple':[[sample['e1_id'],sample['p_id'],sample['ans_id']],[sample['e1'],sample['query'],sample['ans_list']]]
        }
    
    else:
        docs=copy.copy(sample['docs'])
        # doc
        document=[]
        offsets = []
        lemma = []
        pos = []
        ner = []
        raw_can=[]
        for i,doc in enumerate(docs):
            resort_can=sorted(sample[can_name][i], key=lambda x:len(x),reverse=True)     # raw_can
            raw_can.append(resort_can)
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
            document.append(doc_tokens['words'])
            offsets.append(doc_tokens['offsets'])
            lemma.append(doc_tokens['lemma'])
            pos.append(doc_tokens['pos'])
            ner.append(doc_tokens['ner']) 
            
        raw_can_new=[]
        pos_c=[]   # for each doc, record each can's all pos in doc tokens
        for did in range(len(docs)):                             #           c1    c2  c3  c4
            pos_doc=[]                     # this doc's 4 can's all pos  [[1,3,5],[6.7],[],[2]]
            raw_can_doc=[]
            for c in raw_can[did]:
                cc= '@'+'_'.join(c.split())
                pos_cc=[i for i in range(len(document[did])) if document[did][i]==cc]  # can c's all pos in doc i  [1,3,5] / []
                if len(pos_cc)==0:
                    continue
                pos_doc.append(pos_cc)
                raw_can_doc.append(c)
            pos_c.append(pos_doc)
            raw_can_new.append(raw_can_doc)
        
        raw_can=raw_can_new
        
        leave_pos=[i for i in range(len(sample['docs'])) if raw_can[i]!=[]]
        document=[document[i] for i in leave_pos]
        offsets=[offsets[i] for i in leave_pos]
        lemma=[lemma[i] for i in leave_pos]
        pos=[pos[i] for i in leave_pos]
        ner=[ner[i] for i in leave_pos]
        
        pos_c=[pos_c[i] for i in leave_pos]
        raw_can=[raw_can[i] for i in leave_pos]
        
        sample['docs']=[sample['docs'][i] for i in leave_pos]
        sample['candis_links']=[sample['candis_links'][i] for i in leave_pos]
        sample['valid_cands']=[sample['valid_cands'][i] for i in leave_pos]
        sample['candis']=[sample['candis'][i] for i in leave_pos]                                
        sample['phrase_tokens']=[sample['phrase_tokens'][i] for i in leave_pos]    
        sample['tokens']=[sample['tokens'][i] for i in leave_pos]
        sample['Q_tokens']=[sample['Q_tokens'][i] for i in leave_pos]
        sample['Q2anchor']=[sample['Q2anchor'][i] for i in leave_pos]
        sample['Q2title']=[sample['Q2title'][i] for i in leave_pos]
        
        assert [] not in raw_can
        
        Q_id=[]
        Q_name=[]
        Q_desp=[]
        c2Q=[]          # c2Q[i][c_id_in_doc]: doc i's  c's all Q pos in Q_set
        ans_in_Q=[]     # [i]:  ans pos in doc i's Q_set
        ans_in_can=[]   # [i]:  ans mention pos in doc i's can
        ans_exist=False
        for i in range(len(document)):
            Q_set=set()
            for c in raw_can[i]:
                Q=sample['candis_links'][i][c]
                Q_set|=set(tuple([tuple(e) for e in Q]))
            Q_set=list(Q_set)
            Q_id.append([e[0] for e in Q_set])
            Q_name.append([e[1].split() for e in Q_set])
            Q_desp.append([Q_desp_new[e[0]] if e[0] in Q_desp_new else Q_name[-1][idx] for idx,e in enumerate(Q_set)])
            if len(e2_ids)==1:
                if e2_ids[0] in Q_id[-1]:
                    ans_in_Q.append(Q_id[-1].index(e2_ids[0]))
                    ans_in_can.append(raw_can[i].index(c))    # ans_pos in doc i'th cand
                    ans_exist=True
                else:
                    ans_in_Q.append(-1)
                    ans_in_can.append(-1)
            else:
                if any ([e2_id for e2_id in e2_ids if e2_id in Q_id[-1]]):
                    ans_exist=True
                    correct_Q=[e2_id for e2_id in e2_ids if e2_id in Q_id[-1]]
                    ans_in_Q.append(Q_id[-1].index(correct_Q[0]))
                    ans_in_can.append(raw_can[i].index(c))    
                else:
                    ans_in_Q.append(-1)
                    ans_in_can.append(-1)
    
            # mention 2 Q    each can's Q_pos  in Q_id/Q_name/Q_desp 
            c2Q_doc=[]
            for c in raw_can[i]:
                Q=sample['candis_links'][i][c]
                Q_list=list(set(tuple([e[0] for e in Q])))  # mention c's all Q
                c2Q_pos=[Q_id[i].index(e_id) for e_id in Q_list]   # pos in all Q of this doc
                c2Q_doc.append(c2Q_pos)
            c2Q.append(c2Q_doc)
            
        # make process_dataset a generator（生成器) ,each time return a sample
        sample_new= {
            'ans_exist':ans_exist,
            'id': qid,                      # this ques id
            'question': question,           # all ques tokens
            'qlemma': qlemma,               # all ques tokens's lemma
            # all doc's
            'document': document,           # all doc  tokens
            'offsets': offsets,             # all tokens's span in doc
            'lemma': lemma,                 # all doc  tokens's lemma
            'pos': pos,                     # all doc  tokens's pos
            'ner': ner,                     # all doc  tokens's ner
            'raw_can': raw_can,             # n_can: all valid cands
            'pos_c':pos_c,                  # n_doc,n_c,max_can_pos:  all raw cands pos in doc_tokens              
            'Q_id':Q_id,                    # each Q's id
            'Q_name':Q_name,                # each Q's name (splited)
            'Q_desp':Q_desp,                # each Q's desp (splited)
            'can2Q':c2Q,                    # each can's correspond Q's pos in Q_id/Q_name/Q_desp  [2,3,4],[2,4],[1,2,3]
            'ans_in_can':ans_in_can,        # ans mention pos in can .  can get ans mention tokens span by this pos.
            'ans_in_Q':ans_in_Q,            # ans pos in Q_set .        can get ans information by this pos.  -1/pos
            'ans_name':sample['ans_list'],  # ans_name
            'ans_id':sample['ans_id'],      # ans_id
            'triple':[[sample['e1_id'],sample['p_id'],sample['ans_id']],[sample['e1'],sample['query'],sample['ans_list']]]
        }            
                
    sample['ans_exist']=ans_exist
    new_samples.append(sample)
    new_list.append(sample_new)
#    print(sample_new)
    if ans_exist:
        n_in+=1
#    if p_label in P_dict:
#        if P_dict[p_label]>args.gap:
#            new_list_multi.append(sample_new)
#        elif P_dict[p_label]<=1.05:
#            new_list_single.append(sample_new)

if args.single_token:
    rp=''
else:
    rp='_rep'


#new_folder='data/final_test_50/'
# no_e_ndoc50_raw.json
# dev_contain_e_ndoc50_raw.json

output_file=new_folder+'{}_contain_e'+rp+'_'+can_name+'_tokenized_all.json'.format(args.mode) if args.contain \
       else new_folder+'no_e'+rp+'_'+can_name+'_tokenized_all.json'.format(args.mode)
       #  'dev_contain_e_valid_cands_tokenized_all.json'/'dev_contain_e_candis_tokenized.json'
       #  no_e_rep_valid_cands_tokenized_all.json
with open(output_file,'w') as f:
    json.dump(new_list,f)

if args.contain:
    output_file=new_folder+'{}{}_contain_e.json'.format(args.mode,rp)  # dev_rep_contain_e.json/dev_contain_e.json
else:
    output_file=new_folder+'no{}_e.json'.format(rp)
with open(output_file,'w') as f:
    json.dump(new_samples,f)

print(args.mode)
print('single_token:',args.single_token)
print('valid_can:',args.valid_can)
print('have ans rate:',n_in/len(new_samples))

#dev
# single_token True
# valid_can True
#have ans rate: 0.7826283646233884 mul_rate: 0.3227776521149061
#test
#single_token: True
#valid_can: True
#have ans rate: 0.7806958473625141 mul_rate: 0.588327721661055



#dev
#single_token: False
#valid_can: True
#have ans rate: 0.7762949558923321 mul_rate: 0.3227776521149061
#test
#single_token: False
#valid_can: True
#have ans rate: 0.7721661054994389 mul_rate: 0.588327721661055