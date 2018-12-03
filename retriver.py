#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 20:08:18 2018

@author: xuweijia
"""
from __future__ import division
import sys
reload(sys) 
sys.setdefaultencoding('utf8')
tool='spacy'
import argparse
import spacy
import numpy as np
from collections import Counter
import json
from utils import entity_groups
nlp_spacy = spacy.load('en')
index_dir='/data/disk2/private/xuweijia/test_lucene/Indexfile_with_link_12.index'
T_db='/data/disk1/private/xuweijia/new_NER_data/t_cleaned_30.db'
raw2Q_file='/data/disk1/private/xuweijia/new_NER_data/raw2Q_expand_30.json'      # raw word 2 Q  [[e_id,e_label],...]   (incluude all disam in triple entity  + Qlabel2id  multi expand )
Qlabel_dict_file='/data/disk1/private/xuweijia/new_NER_data/elabel2id_e30.json'  # e_label 2 e_id (list)  a_ext
def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')
parse = argparse.ArgumentParser(description='call retriver')
parse.register('type', 'bool', str2bool)
parse.add_argument('--input_test_file', type=str, help='.../test_file,already contain triples')
parse.add_argument('--topK_sens',  type=int, default=20,help='retrive topK sens each sample')
parse.add_argument('--output_test_file', type=str, help='output current statistics') # all tmp
parse.add_argument('--get_new_ans', type='bool', default=False,help='retrive new e2 or not')
args = parse.parse_args()

import lucene,time
import nltk
from java.io import File
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.util import Version
lucene.initVM(vmargs=['-Djava.awt.headless=true'])
analyzer = StandardAnalyzer(Version.LUCENE_CURRENT)
directory=FSDirectory.open(File(index_dir))
searcher = IndexSearcher(DirectoryReader.open(directory))
parser=QueryParser(Version.LUCENE_CURRENT, "sentence", analyzer)

english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%',"''",'``',"'s","-","--",'–']
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(english_punctuations)
stopwords.remove('by')

with open(raw2Q_file,'r') as f:
    raw2Q=json.load(f)
with open(Qlabel_dict_file,'r') as f:
    Qlabel_dict=json.load(f)

with open(args.input_test_file,'r') as f:
    samples=json.load(f)
    
import sqlite3
t_conn=sqlite3.connect(T_db)
t_cursor=t_conn.cursor()
select_t='select t_id,t_label from triples WHERE h_id=? AND p_id=?'

top_K_sentence=args.topK_sens
start=time.time()
count_SP=dict()  # avoid repeated e1,p
new_samples=[]
n=0
n_s=0
# if new, delete repeat with train
for idx,sample in enumerate(samples):
    # or just use old triples ans,
    s_label,p_label=sample['e1'],sample['query']
    e1_id,p_id=sample['e1_id'],sample['p_id']
    if args.get_new_ans:
        t_cursor.execute(select_t,(e1_id,p_id,))
        ts=t_cursor.fetchall()
        if len(ts) !=0:
            ans_id= [t[0] for t in ts]
            ans= [t[1] for t in ts]
        else:
            continue
    else:
        ans=sample['ans_list'] if isinstance(sample['ans_list'],list) else [sample['ans_list']]
        ans_id=sample['ans_id'] if isinstance(sample['ans_list'],list) else [sample['ans_id']]  
    # have s,p, merge ans ,continue / 
    if count_SP.get((e1_id,p_id)):
        ans_old,ans_id_old,idx_old=count_SP.get((e1_id,p_id))
        if set(ans_id_old)==set(ans_id) : # old ans contain this
            continue
        else:
            added_pos=[ans_id.index(a_id) for a_id in ans_id if a_id not in ans_id_old]
            ad_ans=[ans[i] for i in range(len(ans_id)) if i in added_pos]
            ad_ans_id=[ans_id[i] for i in range(len(ans_id)) if i in added_pos]
            new_samples[idx_old]['ans_list'].extend(ad_ans)
            new_samples[idx_old]['ans_id'].extend(ad_ans_id)   
            continue
    else:
        count_SP[(e1_id,p_id)]=(ans,ans_id,len(new_samples))
    # retrive
    query_str=' ({}) AND ( OR {}) '.format(s_label,p_label)
    try:
        query = parser.parse(QueryParser.escape(query_str))
    except:
        continue
    Tophit=searcher.search(query,top_K_sentence)
    scoreDocs = Tophit.scoreDocs
    result=[]
    for scoreDoc in scoreDocs:
        # print " Score:", scoreDoc.score
        doc = searcher.doc(scoreDoc.doc)
        sen_dic={}
        sen_dic["title"]=          doc.get("title")
        sen_dic["sen"]=            doc.get("sentence")
        sen_dic["link_sentence"] = doc.get("link_sentence")
        sen_dic["Q_tokens"] =      json.loads(doc.get("Q_tokens"))
        sen_dic["phrase_tokens"] = json.loads(doc.get("phrase_tokens"))
        sen_dic["Q2anchor"] =      json.loads(doc.get("Q_dict"))
        sen_dic["Q2title"] =       json.loads(doc.get("Q_title"))
        result.append(sen_dic)
    if len(result)==0:
        query_str=' ({}) OR ( OR {}) '.format(s_label,p_label)
        query = parser.parse(QueryParser.escape(query_str))
        #query = parser.parse(query_str)
        Tophit=searcher.search(query,top_K_sentence)
        scoreDocs = Tophit.scoreDocs
        for scoreDoc in scoreDocs:
            # print " Score:", scoreDoc.score
            doc = searcher.doc(scoreDoc.doc)
            sen_dic={}
            sen_dic["title"]=          doc.get("title")
            sen_dic["sen"]=            doc.get("sentence")
            sen_dic["link_sentence"] = doc.get("link_sentence")
            sen_dic["Q_tokens"] =      json.loads(doc.get("Q_tokens"))
            sen_dic["phrase_tokens"] = json.loads(doc.get("phrase_tokens"))
            sen_dic["Q2anchor"] =      json.loads(doc.get("Q_dict"))
            sen_dic["Q2title"] =       json.loads(doc.get("Q_title"))
            result.append(sen_dic)
    # deal with docs
    if len(result)==0: # no sens with s,p or
        continue

    triple_old_candidates=[]
    triple_valid_candidates=[]
    
    triple_all_sentences=[]
    triple_all_candidates=[]
    triple_all_candidate_links=[]
    triple_link_flags=[]
    triple_all_flags=[]
    triple_new_tokens=[]
    
    triple_link_sens=[]
    triple_Q_tokens=[]
    triple_phrase_tokens=[]
    triple_Q_dict=[]
    triple_Q_title=[]
    
    for sample_dict in (result):
        sen=sample_dict["sen"]
        Q2anchor=sample_dict["Q2anchor"]
        Q2title=sample_dict["Q2title"]
        anchor2Q=dict(zip(Q2anchor.values(),Q2anchor.keys()))
    
        doc = nlp_spacy(sen)
        entities=[]
        for token in doc:
            entities.append((token.text,token.ent_type_))
        tokens,NER_tokens,old_candidates,NER_candidates=entity_groups(entities,tool)
    
        # can must often, 
        candidates=[c for c in old_candidates if Qlabel_dict.get(c)!=None or c in raw2Q]     # Qlabel_dict's multi  all in raw2Q
        #candidates.extend(anchor2Q)
        candidates=list(set(candidates))
    
        # candidates=[c for c in candidates if Qlabel_dict.get(c)!=None] 
        candi_link={}
        # can in raw2Q
        disambiguations=[c for c in candidates if c in raw2Q]               # include all multi (label --multi e_id)
        if len(disambiguations)!=0:
            linked_candidates=[]
            for d in disambiguations:
                linked_candidate=raw2Q[d]                     # [['Q155965', 'Cypriot'], ['Q162031', 'Danish 1st Division']]
                linked_Q=[Q for Q,label in linked_candidate]
                linked_candidates.extend(linked_Q)
                candi_link[d]=raw2Q[d]
        else:
            linked_candidates=[]
        # can not in raw2Q, must in Qlabel_dict, and single
        single_c=[c for c in candidates if c not in disambiguations]
        if len(single_c)!=0:
            for c in single_c:
                e_id=Qlabel_dict[c][0]
                candi_link[c]=[[e_id,c]]
                linked_candidates.append(e_id)
        # 3
        for c,e_id in anchor2Q.items():
            if candi_link.get(c) !=None:
                if [e_id,Q2title[e_id]] not in candi_link[c]:
                    candi_link[c].append([e_id,Q2title[e_id]])
                    linked_candidates.append(e_id)
            else:
                candi_link[c]=[[e_id,Q2title[e_id]]]
                linked_candidates.append(e_id)
    
       # correct candidate links/ valid candidates
       # 1 fix unvalid mention name
        flag_del=False
        M=[]
        for m in candi_link:
            if m.strip()!=m and m.strip() in candi_link:
                M.append(m)
                Q_m=candi_link[m]
                Q_m= [Q for Q in Q_m if Q_m not in candi_link[m.strip()]]
                candi_link[m.strip()].extend(Q_m)
                flag_del=True
            elif m.strip()!=m:
                Q_m=candi_link.pop(m)
                candi_link[m.strip()]=Q_m
            if m.strip('"')!=m and m.strip('"') in candi_link:
                M.append(m)
                Q_m=candi_link[m]
                Q_m= [Q for Q in Q_m if Q_m not in candi_link[m.strip('"')]]
                candi_link[m.strip('"')].extend(Q_m)
                flag_del=True 
        if flag_del:
            for m in M:
                candi_link.pop(m)
        # 2 del one Q's multi name            
        for k in candi_link:
            v=candi_link[k]
            Q_id=[e[0] for e in v]
            Q_name=[e[1] for e in v]
            if len(set(Q_id))!=len(Q_id):
                multi_Qid=[kk for kk ,vv in dict(Counter(Q_id)).items() if vv>1]
                drop_index=[]
                for e_id in multi_Qid:
                    pos=np.where(np.array(Q_id)==e_id)[0]
                    multi_name=list(np.array(Q_name)[pos])
                    if any([a for a in ans if a in multi_name]):
                        # drop other name have same e_id
                        drop_index.extend([i for i in pos if Q_name[i] not in ans])
                    else:
                        # shortest name
                        leng=[len(name) for name in multi_name]
                        keep_name=multi_name[leng.index(min(leng))]
                        drop_index.extend([i for i in pos if Q_name[i]!=keep_name])
                candi_link[k]= [m for m in v if v.index(m) not in drop_index]

        triple_old_candidates.append(old_candidates)
        valid_candidates=list(candi_link.keys())               # candi with anchors
        triple_valid_candidates.append(valid_candidates)

        triple_all_candidates.append(candidates)               # just NER, with Q candidates
        triple_all_candidate_links.append(candi_link)
        triple_all_sentences.append(sen)
        triple_new_tokens.append(tokens)
        #  NER_text='@_'+'_'.join(candidate)
        # 4
        triple_link_sens.append(sample_dict["link_sentence"])
        triple_Q_tokens.append(sample_dict["Q_tokens"])
        triple_phrase_tokens.append(sample_dict["phrase_tokens"] )
        triple_Q_dict.append(sample_dict["Q2anchor"])
        triple_Q_title.append(sample_dict["Q2title"])

    len_s=len(triple_all_sentences)
    if len_s >0:
        n+=1
        n_s+=len_s
        end=time.time()
        # print('n_dev:{},n_dev_s:{},time={:.3f}'.format(n,n_s,(end-start)/3600))
        sample_new=dict()
        sample_new['e1']=s_label
        sample_new['e1_id']=e1_id
        sample_new['query']=p_label
        sample_new['p_id']=p_id
        sample_new['ans_list']=ans
        sample_new['ans_id']=ans_id                             
        sample_new['docs']= triple_all_sentences
        sample_new['candis']= triple_all_candidates
        sample_new['orig_cands']=triple_old_candidates
        sample_new['valid_cands']=triple_valid_candidates

        sample_new['candis_links']=triple_all_candidate_links
        sample_new['tokens']= triple_new_tokens
        sample_new['r_des']=sample['r_des']

        sample_new['Q_tokens']=    triple_Q_tokens
        sample_new['phrase_tokens']=triple_phrase_tokens
        sample_new['Q2anchor']=     triple_Q_dict
        sample_new['Q2title']=      triple_Q_title
       # all candidate of s,p
        can_set=set()
        for i in range(len_s):
            can_set|=set(triple_all_candidates[i])   # all candi in s,p
        sample['cans_set']=list(can_set)
        # ans_exist  not know, until tokenize
        print json.dumps(sample_new)
        new_samples.append(sample_new)

## store retrive file  test batch size >all samples
with open(args.raw_test_file,'w') as f:
    json.dump(new_samples,f)


