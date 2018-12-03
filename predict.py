#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 19:48:31 2018

@author: xuweijia
"""
import os
import json
import argparse
import subprocess
import numpy as np
from drqa import raw_predict

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')
# parser
parser = argparse.ArgumentParser()
parser.register('type', 'bool', str2bool)
# with doc dataset
parser.add_argument('--model_file', type=str, default='model_dir/model_span.pkl',help="model_span.pkl/model_contain.pkl/model_NER.pkl/model_string_match_base_dis.pkl  Path to Reader model")
parser.add_argument('--train_mode', type=str, default='span',help='string_match,string_match_base_dis,span,contain,NER')
parser.add_argument('--mode',type=str,default='dev',help='dev/test')
parser .add_argument('--normalize_s', type='bool', default=False,help='norm on init s.')
parser .add_argument('--normalize_q', type='bool', default=False,help='norm on B_max_Q.')
parser .add_argument('--normalize_ss', type='bool', default=False,help='norm on span/ c s tokens.')
parser .add_argument('--multi_dataset', type='bool', default=False,help='all/multi')
parser .add_argument('--single_dataset', type='bool', default=False,help='all/multi')
parser .add_argument('--zero_shot', type='bool', default=False,help='bool')
parser. add_argument('--pool', type='bool', default=False,help='no pool during predict')

parser.add_argument('--gpu',type=int,default=0,help='int')
parser.add_argument('--predict_from_raw',type='bool',default=False,help='use triples retrive docs , get candis, preprocess')
parser.add_argument('--input_file',type=str,default='')
parser.add_argument('--n_docs', type=int, default=20,help="Number of docs to retrieve per query") 
parser.add_argument('--get_new_ans', type='bool', default=False,help='retrive new e2 or not')
parser.add_argument('--use_embedding', type='bool', default=False)
parser.add_argument('--embedding_file', type=str, default='/data/disk1/private/xuweijia/DrQA/data/embeddings/glove.840B.300d.txt')
parser.add_argument('--query_batch_size', type=int, default=3000,help='Query batching size')
parser.add_argument('--batch-size', type=int, default=16,help='Doc batching size')
parser.add_argument('--data_workers', type=int, default=0,help='None:all Number of CPU processes (for tokenizing, etc)')
parser.add_argument('--cuda', type='bool', default=True,help='Train on GPU.')


parser.add_argument('--rerank',type='bool',default=False)  
parser.add_argument('--transX',type=str,default='transE')          # all reranking
parser.add_argument('--epoch' ,type=int,default=100000,help='which epoch')
parser.add_argument('--size',type=int,default=100,help='what size to use to rerank')
parser.add_argument('--rerank_compare',type='bool',default=False,help='compare rerank with no rerank,output file') 
# 1
parser.add_argument('--rerank_method',type=str,default='hard',help='soft/hard') 
parser.add_argument('--rerank_softnormal',type='bool',default=False,help='when soft rerank, if normal score') 

args = parser.parse_args()

if args.mode=='dev':
    mode='dev'
else:
    mode='test'
    
last='all'
if args.multi_dataset:
    last='multi'
if args.single_dataset:
    last='single'


if not args.zero_shot:
    if args.train_mode!='NER':
        input_file='data/final_test/{}_contain_e_valid_cands_tokenized_{}.json'.format(mode,last)
    else:
        input_file='data/final_test/{}_contain_e_rep_valid_cands_tokenized_{}.json'.format(mode.last) 
else:
    if args.train_mode!='NER':
        input_file='data/final_test/no_e_valid_cands_tokenized_{}.json'.format(last)
    else:
        input_file='data/final_test/no_e_rep_valid_cands_tokenized_{}.json'.format(last)
# no ner   dev
#args.dev_file='data/final_test/dev_contain_e_valid_cands_tokenized_all.json' 
           #   'data/final_test/dev_contain_e_valid_cands_tokenized_mul.json'
           #   'data/final_test/dev_contain_e_valid_cands_tokenized_sing.json'
# no ner test
#args.test_file='data/final_test/test_contain_e_valid_cands_tokenized_all.json'
            
# rep
#args.dev_file='data/dev_contain_e_rep_valid_cands_tokenized_all.json'
           #   'data/final_test/dev_contain_e_rep_valid_cands_tokenized_sing.json'
           #   'data/final_test/dev_contain_e_rep_valid_cands_tokenized_mul.json '
#args.test_file='data/test_contain_e_rep_valid_cands_tokenized_all.json'
           
# zero_shot     'data/final_test/no_e_rep_valid_cands_tokenized_all.json'
#               'data/final_test/no_e_valid_cands_tokenized_all.json'
           
           
input_file=args.input_file
#input_file=input_file

if args.predict_from_raw:
    subprocess.call(['mkdir', '-p', 'new_data'])    # os.system('cd /usr/local && mkdir aaa.txt')  / os.system('cd /usr/local ; mkdir aaa.txt')
    input_sample_dir='data/{}_contain_e.json'.format(mode) if args.train_mode!='NER' else 'data/{}_rep_contain_e.json'.format(mode)
    # old_sample_dir=
    output_test_file='new_data/new_{}.json'.format(input_sample_dir.split('/')[-1].split('.')[0])   # 'new_data/new_dev_contain_e.json'

    # 1 retrive    (different number of docs compare with input_test_files)
    list_ex0=os.popen('~/anaconda2/bin/python2.7 retriver.py --input_test_file {} --topK_sens {} --output_test_file {} --get_new_ans {}'.\
                      format(input_sample_dir,args.n_docs,output_test_file,args.get_new_ans)).readlines()           # return command line's results (like print, diff line when print in different time)
    list_ex=[json.loads(line.strip()) for line in list_ex0]                                                    # using json dump to print each sample, then load
    print('len list_ex:{}'.format(list_ex))
    
    # 2 test        (and get statistocs file in new_data file)
    valid_sample=True   
    information=os.popen('python test_test_samples_def.py --test_file {} ----valid_sample {} --mode {}'.\
                      format(output_test_file,valid_sample,mode)).readlines()
    #print (information) 
         
    # 3 tokenize    (if valid can, if NER(others same))
    if args.train_mode=='NER':
        single_token=False
    else:
        single_token=True
        
    os.system('python preprocess_test_def.py --input_file {} --single_token {} --valid_can {} --gap {} --mode {}'.\
              format(output_test_file,single_token,valid_sample,1.5,mode))

import torch
args.cuda = args.cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpu)

with open(input_file,'r') as f:
    samples=json.load(f)
    if not args.rerank:
        raw_predict(samples,args,args.model_file,args.use_embedding,args.embedding_file,args.cuda,args.normalize_q,args.normalize_s,args.normalize_ss,rerank_info=None)
    else:
        rerank_info=dict()
        rerank_info['E']=45128
        rerank_info['R']=78
        # file name
        if args.size==100:
            rerank_info['e_embed']='transX/{}/embedding_wiki_{}/entity2vec.bin'.format(args.transX,args.epoch)
            rerank_info['r_embed']='transX/{}/embedding_wiki_{}/relation2vec.bin'.format(args.transX,args.epoch)
            if args.transX!='transE':
                rerank_info['A_embed']='transX/{}/embedding_wiki_{}/A.bin'.format(args.transX,args.epoch)
        else:
            rerank_info['e_embed']='transX/{}/embedding_wiki_{}_{}/entity2vec.bin'.format(args.transX,args.epoch,args.size)
            rerank_info['r_embed']='transX/{}/embedding_wiki_{}_{}/relation2vec.bin'.format(args.transX,args.epoch,args.size)
            if args.transX!='transE':
                rerank_info['A_embed']='transX/{}/embedding_wiki_{}_{}/A.bin'.format(args.transX,args.epoch,args.size) 
        e_id2idx='transXe_id2idx.json'
        p_id2idx='transX/r_id2idx.json'
        with open(e_id2idx,'r') as f:
            rerank_info['eid2idx']=json.load(f)
        with open(p_id2idx,'r') as f:
            rerank_info['pid2idx']=json.load(f)
        raw_predict(samples,args,args.model_file,args.use_embedding,args.embedding_file,args.cuda,args.normalize_q,args.normalize_s,args.normalize_ss,rerank_info=rerank_info)

        
        
        

        
