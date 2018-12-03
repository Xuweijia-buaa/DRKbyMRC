#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 16:14:19 2018

@author: xuweijia
"""
import argparse
import subprocess
import torch
import os
import logging
import sys
import numpy as np
import json
from utils import build_feature_dict,build_word_dict, ReaderDataset,SortedBatchSampler, top_question_words, Timer,AverageMeter,isnan
from Vectorize import batchify
from Reader import DocReader
from predict_during_train import get_test_loader,get_test_result,predict
import multiprocessing
logger = logging.getLogger()
parser = argparse.ArgumentParser('Document Reader',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')

parser.register('type', 'bool', str2bool)
# 为原始的parser添加新的属性组，添加后重新用arg=parser.parse_args()得到的arg，包含所有新属性args.cuda
# Files
files = parser.add_argument_group('Filesystem')
files.add_argument('--train_mode', type=str, default='contain',help='string_match_base_dis,string_match_base_dis,span,contain,NER')   # ~
files.add_argument('--Q_mode', type=str, default='Q_desp',help='Q_name,Q_desp,Q_all')                                           # ~
files.add_argument('--m_loss', type='bool', default=False,help='whether use mention loss in mode span,contain,NER')       # ~
files.add_argument('--db_softmax', type='bool', default=False,help='whether use softmax over f')
files.add_argument('--spa', type=float, default=1,help='sparcity')
files.add_argument('--test_doc_top_1', type='bool', default=True,help='each doc predcit 1')
files.add_argument('--model_dir', type=str, default='model_dir',help='Directory for saved models/checkpoints/logs')
files.add_argument('--train_file', type=str,default='/home/xuweijia/my_drqa/train_before_statis_new.json', help='Preprocessed train file')
files.add_argument('--dev_file', type=str,default='dev_file_after_preprocess.json',help='Preprocessed dev file')
files.add_argument('--test_file', type=str,default='dev_file_after_preprocess.json',help='Preprocessed test file')
files.add_argument('--using_embedding', type='bool', default=True,help='whether use embed')
files.add_argument('--test_using_embedding', type='bool', default=False,help='whether use embed')
files.add_argument('--embedding_file', type=str, default='/data/disk1/private/xuweijia/DrQA/data/embeddings/glove.840B.300d.txt',help='Space-separated pretrained embeddings file')
# Runtime environment
runtime = parser.add_argument_group('Environment')
runtime .add_argument('--normalize_q', type='bool', default=True)
runtime .add_argument('--normalize_s', type='bool', default=True)
runtime .add_argument('--test_normalize_q', type='bool', default=False)
runtime .add_argument('--test_normalize_s', type='bool', default=False)
runtime.add_argument('--no-cuda', type='bool', default=False,help='Train on CPU, even if GPUs are available.')
runtime.add_argument('--gpu', type=int, default=0,help='Run on a specific GPU')
runtime.add_argument('--data_workers', type=int, default=0,help='Number of subprocesses for data loading')
runtime.add_argument('--parallel', type='bool', default=False,help='Use DataParallel on all available GPUs')
runtime.add_argument('--random-seed', type=int, default=1013, help=('Random seed for all numpy/torch/cuda ''operations (for reproducibility)'))
runtime.add_argument('--num_epochs', type=int, default=40,help='Train data iterations')  # epoch
runtime.add_argument('--batch-size', type=int, default=64,help='Batch size for training')                                   # ~
runtime.add_argument('--test_batch_size', type=int, default=16,help='Batch size during validation/testing')
# Saving + loading
save_load = parser.add_argument_group('Saving/Loading')
save_load.add_argument('--checkpoint', type='bool', default=False,help='Save model + optimizer state after each epoch')
save_load.add_argument('--pretrained', type=str, default='', help='pretrained model to warm-start with')
save_load.add_argument('--expand_dictionary', type='bool', default=False,help='add words (not in old model.word_dict) into word_dict(new index, new-embedding randomly initialized))') 
# Data preprocessing
preprocess = parser.add_argument_group('Preprocessing')
preprocess.add_argument('--uncased-question', type='bool', default=False,help='Question words will be lower-cased')
preprocess.add_argument('--uncased-doc', type='bool', default=False,help='Document words will be lower-cased')
preprocess.add_argument('--restrict_vocab', type='bool', default=True, help='Only use words in embedding_file') 
# General
general = parser.add_argument_group('General')
general.add_argument('--display_iter', type=int, default=25,help='Log state after every <display_iter> epochs')
general.add_argument('--sort-by-len', type='bool', default=True,help='Sort batches by length for speed')
# Model architecture
model = parser.add_argument_group('DrQA Reader Model Architecture') 
model.add_argument('--model-type', type=str, default='rnn',help='Model architecture type')
model.add_argument('--rnn_type', type=str, default='lstm',help='RNN type: LSTM, GRU, or RNN')
model.add_argument('--doc_layers', type=int, default=3,help='Number of encoding layers for document')
model.add_argument('--ques_layers', type=int, default=3,help='Number of encoding layers for question')
model.add_argument('--Q_layers', type=int, default=3,help='Number of encoding layers for Q desp/Q_name')
model.add_argument('--concat_layers', type='bool', default=True, help='Combine hidden states from each encoding layer')
model.add_argument('--doc_use_qemb', type='bool', default=True,help='Whether to use soft attention (weighted question embeddings) to encode D')
model.add_argument('--q_self_weight', type='bool', default=True,  help='self attention to encode question,get question representation') 
model.add_argument('--embedding_dim', type=int, default=300,help='Embedding size if embedding_file is not given/ if given, same to embedding_file')
model.add_argument('--hidden_size', type=int, default=128,help='Hidden size of RNN units')    
 # features to encode doc tokens  (pos,ner,tf,lemma)
features = parser.add_argument_group('DrQA Reader Model Details')
features.add_argument('--use-in-question', type='bool', default=True,help='Whether to use  # if q word in D feature to encode D')  
features.add_argument('--use-pos', type='bool', default=True,help='Whether to use pos features')                         
features.add_argument('--use-ner', type='bool', default=True,help='Whether to use ner features')
features.add_argument('--use-lemma', type='bool', default=True,help='Whether to use lemma features')
features.add_argument('--use-tf', type='bool', default=True,help='Whether to use term frequency features')
# Optimization details
optim = parser.add_argument_group('DrQA Reader Optimization')
optim.add_argument('--dropout_emb', type=float, default=0.4, help='input (word-embedding) dropout')
optim.add_argument('--dropout_rnn', type=float, default=0.4, help='Dropout rate for RNN hidden states')
optim.add_argument('--dropout_rnn_output', type=float, default=0.4,help='Whether to dropout the RNN output')
optim.add_argument('--optimizer', type=str, default='adamax',help='Optimizer: sgd/adam/adamax')
optim.add_argument('--learning-rate', type=float, default=0.1,help='Learning rate for SGD only  lr 0.1')
optim.add_argument('--grad-clipping', type=float, default=10, help='Gradient clipping')
optim.add_argument('--weight-decay', type=float, default=0, help='Weight decay factor')
optim.add_argument('--momentum', type=float, default=0, help='Momentum factor')
optim.add_argument('--fix_embeddings', type='bool', default=False, help='fix pretained input word_embedding, no one tuned / only given embedding_file/pretrina_model')
optim.add_argument('--tune_partial', type=int, default=1000, help='top N question word embeddings are tuned 2018,tune all')
optim.add_argument('--rnn_padding', type='bool', default=True,help='Explicitly account for padding in RNN encoding')
optim.add_argument('--pool', type='bool', default=False,help='no pool during predict')
optim.add_argument('--max-len', type=int, default=15,help='The max span (token level) allowed during decoding')

args = parser.parse_args()
subprocess.call(['mkdir', '-p', args.model_dir])    # 
args.model_file=os.path.join(args.model_dir,'model_{}_mloss{}_db{}_{}_tune{}_pad{}_batch{}_drop{}_restrictV{}_spa{}'.format(args.train_mode,args.m_loss,args.db_softmax,args.Q_mode,args.tune_partial,args.rnn_padding,args.batch_size,args.dropout_rnn,args.restrict_vocab,args.spa) +'.pkl')

args.log_file = os.path.join(args.model_dir, 'model' + '.txt')

if args.tune_partial==0:
    args.fix_embeddings=True
if args.train_mode!='NER':
    if args.spa==1:
        args.train_file='data/final_train/train_tokenized.json'
    else:
        args.train_file='data/final_train/train_tokenized_spa{}.json'.format(args.spa)
    args.dev_file='data/final_test/dev_contain_e_valid_cands_tokenized_all.json'
    args.test_file='data/final_test/test_contain_e_valid_cands_tokenized_all.json'
    
else:
    if args.spa==1:
        args.train_file='data/final_train/train_rep_tokenized.json'
    else:
        args.train_file='data/final_train/train_rep_tokenized_spa{}.json'.format(args.spa)        
    args.dev_file='data/final_test/dev_contain_e_rep_valid_cands_tokenized_all.json'
    args.test_file='data/final_test/test_contain_e_rep_valid_cands_tokenized_all.json'
    
if not args.using_embedding:
    args.embedding_file=None
# Set cuda and gpu
args.cuda = (not args.no_cuda)  and  (torch.cuda.is_available())
if args.cuda:
    torch.cuda.set_device(args.gpu)

# Set random state for np,torch,torch.cuda
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

# Set logging
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                        '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)
if args.log_file:
    if args.checkpoint:
        logfile = logging.FileHandler(args.log_file, 'a')
    else:
        logfile = logging.FileHandler(args.log_file, 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
logger.info('COMMAND: %s' % ' '.join(sys.argv)) #获取运行python文件的时候命令行参数，且以list形式存储参数

# load data
with open(args.train_file,'r') as f:
    train_exs=json.load(f)
    #train_exs=train_exs[:100]

with open(args.dev_file,'r') as f:
    dev_exs=json.load(f)
    #dev_exs=dev_exs[:100]

with open(args.test_file,'r') as f:
    test_exs=json.load(f)
    #test_exs=test_exs[:100]
# build dict
feature_dict = build_feature_dict(args, train_exs) # feature_dict['in_question']=0, ['in_question_uncased']=1,['in_question_lemma']=2,['pos=NN']=3,['pos=IN']=4,['pos=DT']=5,.
word_dict = build_word_dict(args, train_exs , dev_exs+test_exs)
logger.info('Num words = %d' % len(word_dict))

# --------------------------------------------------------------------------
logger.info('-' * 100)
logger.info('Make data loaders')
# single ex vectorized
train_dataset = ReaderDataset(train_exs, args, word_dict, feature_dict, if_train=True)
# sample stategy
if args.sort_by_len:
    train_sampler = SortedBatchSampler(train_dataset.lengths(),
                                            args.batch_size,
                                            shuffle=True)
else:
    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    
train_loader = torch.utils.data.DataLoader(   
    train_dataset,                            # all vectorized train samples
    batch_size=args.batch_size,
    sampler=train_sampler,                    # shuffle must be false if define sampler  (if reshuffled at every epoch) 
    num_workers=args.data_workers,
    collate_fn=batchify,                      # merges a list of samples to form a mini-batch. give a batch, return a 
    pin_memory=args.cuda,                     # 内存充足的时候，可以设置pin_memory=True  (pin_memory=True可以使DataLoader将batch返回到固定内存中)  副本来自固定（页锁）内存时，主机到GPU的复制速度要快很多
)


dev_examples,dev_Dataset,dev_loader=get_test_loader(dev_exs,args,word_dict,feature_dict,args.cuda)
                                         
# -------------------------------------------------------------------------
# PRINT CONFIG
#logger.info('-' * 100)
#logger.info('CONFIG:\n%s' %json.dumps(vars(args), indent=4, sort_keys=True))

# --------------------------------------------------------------------------
# model
model = DocReader(args, word_dict, feature_dict)
if args.embedding_file:
    model.load_embeddings(word_dict.tokens(), args.embedding_file)

if args.tune_partial > 0: # Set up partial tuning of embeddings
    logger.info('-' * 100)
    logger.info('Counting %d most frequent question words' %args.tune_partial)  # return most common question words triple  [('e', 2), ('ve', 1)]
    if args.tune_partial==2018:
        args.tune_partial=len(word_dict)
    top_words = top_question_words(args, dev_exs, model.word_dict)            # word must in current word_dict  (scrath from init / expand dic)
    #for word in top_words[:5]:
    #    logger.info(word)
    # logger.info('...')
    #for word in top_words[-6:-1]:
    #    logger.info(word)
    model.tune_embeddings([w[0] for w in top_words]) # all tuned top words in question. tuned, put in fist 2:2+args.tune / others fixed. also change model.'word_dict','embedding'
model.init_optimizer()  # optimizer: sgd/adamax
if args.cuda:
    model.cuda()
if args.parallel:
    model.parallelize()  # torch.nn.DataParallel(self.network),use multiple GPUs

start_epoch = 0
logger.info('-' * 100)
#logger.info('Starting training...')
stats = {'timer': Timer(), 'epoch': 0, 'best_valid': 0}

# train start 
# args.num_epochs=2
for epoch in range(start_epoch, args.num_epochs):               # epoch from last checkpoint/ 0
    stats['epoch'] = epoch
    # Train. 2 classifier loss
    train_loss = AverageMeter()  # count_class
    epoch_time = Timer()         # time class
    # one epoch
    model.set_normalize(args.normalize_q,args.normalize_s)
    for idx, ex in enumerate(train_loader):   
        # loss,B
        loss,B=model.update(ex) 
        #final_Q,loss,answear,span2c,span_normal,span_start,span_end,span_s,span_mask,score_e,score_s,pure_Q,B_max_c,ans_in_can,Q_mask,CQ_mask,ques_final,doc_output,dw_mask,\
        #       score_s_old, score_e_old,pure_Q_old,ques_final_old,doc_output_old,span_mask_old,span_start_old,span_end_old,span_s_old,span_normal_old,B_max_c_old,pure_Q_old2,B_max_Q_old\
        #       =model.update(ex)
        # span2c,span_normal,span_start,span_end,span_s,span_mask,score_s_old, score_e_old,score_e,score_s,pure_Q,B_max_c,ans_in_can,Q_mask,CQ_mask,ques_final,doc_output,dw_mask=model.update(ex)
        # score_e,score_s,target_e,target_s,final_Q,pure_Q,B_max_Q,B_max_Q_old1,B_max_Q_old2,ans_in_can,ans_index,answear,predict_prob,loss1,loss2,loss,Q_mask,CQ_mask=model.update(ex)
        #assert torch.sum(isnan(score_e.data.cpu()))==0
        #assert torch.sum(isnan(score_s.data.cpu()))==0
        #assert torch.sum(isnan(B_max_c.data.cpu()))==0
        #assert torch.sum(isnan(final_Q.data.cpu()))==0
        assert loss>=0
        assert loss<100000000
        train_loss.update(loss,B)
        if idx % args.display_iter == 0:
            logger.info('train: Epoch = %d | iter = %d/%d | ' %(stats['epoch'], idx, len(train_loader)) +'loss = %.2f | elapsed time = %.2f (s)' %(train_loss.avg, stats['timer'].time()))
            train_loss.reset()
    logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %(stats['epoch'], epoch_time.time()))

    # dev and  Save best valid
    model.set_normalize(args.test_normalize_q,args.test_normalize_s)
    
    #model.set_normalize(True,True)
    Q_scores_multi,Q_scores_single,Q_scores_most,Q_scores_random,avg_Q=\
    get_test_result(dev_exs,model,args,dev_loader)
    
    
    exact_match_rate_s,exact_match3_s,exact_match10_s,ex_ans_exist_s,n_ans_exist=predict(dev_exs,Q_scores_single)    
    exact_match_rate_m,exact_match3_m,exact_match10_m,ex_ans_exist_m,_=predict(dev_exs,Q_scores_multi)  
    
    exact_match_rate_most,exact_match3_most,exact_match10_most,ex_ans_exist_most,_=predict(dev_exs,Q_scores_most)   
    exact_match_rate_rand,exact_match3_rand,exact_match10_rand,ex_ans_exist_most,_=predict(dev_exs,Q_scores_random)
       
    exact_match= exact_match_rate_s if exact_match_rate_s>=exact_match_rate_m else exact_match_rate_m
    exact_match_ans= ex_ans_exist_s if ex_ans_exist_s>=ex_ans_exist_m else ex_ans_exist_m
    winning_method= 'single' if exact_match_rate_s>=exact_match_rate_m else 'multi'
    
    if exact_match > stats['best_valid']:
        logger.info('Best valid: %s = %.2f (epoch %d, %d updates),method: %s' %
                    (args.train_mode, exact_match,
                     stats['epoch'], model.updates,winning_method))
        model.save(args.model_file)
        stats['best_valid'] = exact_match
        
    print({'best  valid:': stats['best_valid'] })
    print({'winning_method:':winning_method},{'train_mode':args.train_mode},{'spa':args.spa},{'m_loss':args.m_loss},{'Q_mode':args.Q_mode},\
          {'db_softmax':args.db_softmax},{'tune':args.tune_partial},{'rnn_padding':args.rnn_padding})
    print({'exact_match %': exact_match},{'total:':len(dev_exs)})
    print({'exact_match_ans %': exact_match_ans},{'total_ans:':n_ans_exist})
    print({'exact_match_m %': exact_match_rate_m},{'exact_match_s %': exact_match_rate_s})
    print({'exact_match_m_ans %': ex_ans_exist_m},{'exact_match_s_ans %': ex_ans_exist_s})
    print({'exact_match_m %': exact_match_rate_m},{'exact_match_s %': exact_match_rate_s})
    print({'exact_match3_m': exact_match3_m},{'exact_match_3_s': exact_match3_s})
    print({'exact_match10_m': exact_match10_m},{'exact_match_10_s': exact_match10_s})
    print({'exact_match_most': exact_match_rate_most})  
    print({'exact_match_random': exact_match_rate_rand})  
    print({'avg_Q': avg_Q})  
    print({'ans_exist_rate:':n_ans_exist/len(dev_exs)})
