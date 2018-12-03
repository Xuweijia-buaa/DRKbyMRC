#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 10:52:15 2018

@author: xuweijia
"""
#import unicodedata
import logging
from spacy_tokenizer import SpacyTokenizer
import numpy as np
import torch
from torch.autograd import Variable
from Vectorize import vectorize
logger = logging.getLogger(__name__)#初始化

def to_var(np_input,use_cuda,evaluate=False):
    # if evaluate, volatile=True, no grad be computed/    some need .long()/.float()    requires_grad=True
    if use_cuda:
        output=Variable(torch.from_numpy(np_input),volatile=evaluate).cuda()
    else:
        output=Variable(torch.from_numpy(np_input),volatile=evaluate)
    return output

def to_vars(np_inputs,use_cuda,evaluate=False):
    return [to_var(np_input,use_cuda,evaluate) for np_input in np_inputs]
 
def to_var_torch(torch_input,use_cuda,evaluate=False):
    # if evaluate, volatile=True, no grad be computed/    some need .long()/.float()    requires_grad=True
    if use_cuda:
        output=Variable(torch_input,volatile=evaluate).cuda()
    else:
        output=Variable(torch_input,volatile=evaluate)
    return output
def to_vars_torch(torch_inputs,use_cuda,evaluate=False):
    return [to_var_torch(torch_input,use_cuda,evaluate) for torch_input in torch_inputs]


def isnan(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("The argument is not a tensor", str(tensor))
    return tensor != tensor
    

def find_answer(offsets, begin_offset, end_offset):
    """Match token offsets with the char begin/end offsets of the answer."""
    # offset:spans get from tokenize  (0,5)/(6,8),(9,10)/.../
    start = [i for i, tok in enumerate(offsets) if tok[0] == begin_offset]   # all token's span start can match ans begin character pos
    end = [i for i, tok in enumerate(offsets) if tok[1] == end_offset]       # all token's span end   can match ans end   character pos
    assert(len(start) <= 1)
    assert(len(end) <= 1)
    # if multi ans in this doc, just consider the one who offer begin_offset, end_offset span as ans
    if len(start) == 1 and len(end) == 1:# only if 2 token exact match ans start and end pos, return these token's idx in doc   [0,2]  (token 0-2)
        return start[0], end[0]          # or one token exact match ans start and end pos, return this token's idx in doc       [2,2]  (token 2)
    
# from s find s1's all char span
def find_span(s,s1):
    span=[]
    start=s.index(s1,0)
    span.append([start,start+len(s1)])
    while s.find(s1,start+1)!=-1:
        start=s.index(s1,start+1)
        span.append([start,start+len(s1)])
    return span

def tokenize2(text):
    """Call the global process tokenizer on the input text."""
    anno_set={'annotators': {'lemma', 'pos', 'ner'}}
    TOK2=SpacyTokenizer(**anno_set)
    tokens = TOK2.tokenize(text)     # a Token object
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
# 1 dictionary class
class Dictionary(object): # object 顶级父类 python中所有的类都是直接或者间接继承自object
    NULL = '<NULL>'
    UNK = '<UNK>'
    START = 2

    @staticmethod
    def normalize(token):
        # NFD表示字符应该分解为多个组合字符表示
        #return unicodedata.normalize('NFD', token)
        return token
    # word_dict(NULL 0, UNK 1)
    def __init__(self):
        self.tok2ind = {self.NULL: 0, self.UNK: 1}  # word_dict  [embedding[0]=0  paddding=0]
        self.ind2tok = {0: self.NULL, 1: self.UNK}  # id2word
    # |V|
    def __len__(self):
        return len(self.tok2ind)
    # get each {w:idx} in word_dict
    def __iter__(self):
        return iter(self.tok2ind)
    # if key in Dictionary: if key in w_dict/index2word dict
    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return self.normalize(key) in self.tok2ind
    # Dictionary[key]: get value of key in w_dict(UNK id 1)/index2word dict(UNK)
    # key need to be normalized
    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, self.UNK)
        if type(key) == str:
            return self.tok2ind.get(self.normalize(key),
                                    self.tok2ind.get(self.UNK))
    # Dictionary[key]=item:  set value of key in w_dict/index2word dict
    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2tok[key] = item
        elif type(key) == str and type(item) == int:
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')
    # add (normalized) token to both w_dict/index2word dict 
    def add(self, token):
        token = self.normalize(token)
        if token not in self.tok2ind:
            index = len(self.tok2ind)     # token index
            self.tok2ind[token] = index
            self.ind2tok[index] = token
    # all valid words in w_dict (list, exclude NULL,UNK) 
    def tokens(self):
        """Return all words in this dictionary, except for special tokens. """
        tokens = [k for k in self.tok2ind.keys()
                  if k not in {'<NULL>', '<UNK>'}]
        return tokens


#  2  feature dict
# feature_dict['in_question']=0, ['in_question_uncased']=1,['in_question_lemma']=2,['pos=NN']=3,['pos=IN']=4,['pos=DT']=5,['ner=ORG']=6,...
def build_feature_dict(args, examples):
    # private func, insert this feature name to feature_dic
    def _insert(feature):
        if feature not in feature_dict:
            feature_dict[feature] = len(feature_dict)
    # if no feature
    feature_dict = {}

    # Exact match features
    if args.use_in_question:
        _insert('in_question')
        _insert('in_question_uncased')
        if args.use_lemma:
            _insert('in_question_lemma')

    # Part of speech tag features (all possible pos)
    if args.use_pos:
        for ex in examples:
            for w in ex['pos']:
                _insert('pos=%s' % w)

    # Named entity tag features  (all possible ner)
    if args.use_ner:
        for ex in examples:
            for w in ex['ner']:
                _insert('ner=%s' % w)

    # Term frequency feature
    if args.use_tf:
        _insert('tf')
    return feature_dict

# 3 return all embedding file words (set)(normalized)
    # get all (normalized) embedding file words (set)
def index_embedding_words(embedding_file):
    words = set()
    with open(embedding_file) as f:
        for line in f:
            w = Dictionary.normalize(line.rstrip().split(' ')[0])
            words.add(w)
    return words

# 4 return all words in all examples (set)   (if args.restrict_vocab, only keep in-embedding-file words)
def load_words(args, examples,test_examples):
    # private func
    if args.train_mode=='NER':
        args.restrict_vocab=False
    def _insert(iterable):
        for w in iterable:
            w = Dictionary.normalize(w)
            if valid_words and w not in valid_words:
                continue
            words.add(w)
    # if restrict_vocab, only keep words in embedding files (valid words)
    if args.restrict_vocab and args.embedding_file:
        logger.info('Restricting to words in %s' % args.embedding_file)
        # all embedding file words set
        valid_words = index_embedding_words(args.embedding_file)
        logger.info('Num words in set = %d' % len(valid_words))
    else:
        valid_words = None
    words = set()
    for ex in examples:
        _insert(ex['question'])
        _insert(ex['document'])
        _insert(ex['raw_can'])
        for i in range(len(ex['Q_id'])):
            _insert(ex['Q_name'][i])
            _insert(ex['Q_desp'][i])
            
    for ex in test_examples:
        _insert(ex['question'])
        for idx,doc in enumerate(ex['document']):
            _insert(doc)
            _insert(ex['raw_can'][idx])
            for i in range(len(ex['Q_id'][idx])):     # each doc's all Q
                _insert(ex['Q_name'][idx][i])
                _insert(ex['Q_desp'][idx][i])
    return words

# 5 return dict class have all ex's token (or just those in embedding_files)
def build_word_dict(args, examples,test_examples):
    # a dictionary class, have in2tok, tok2in
    word_dict = Dictionary()
    # all words in examples add in word_dict (each word have a index, if restrict_vocab, filter non-embedding file words)
    for w in load_words(args, examples,test_examples):
        word_dict.add(w)
    return word_dict

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
class ReaderDataset(Dataset):

    def __init__(self, examples, args, word_dict,feature_dict,if_train):
        self.args=args
        self.examples = examples
        self.if_train =if_train   # train == True
        self.dict=[word_dict,feature_dict]

    def __len__(self):
        return len(self.examples)
    
    # vectorized Dataset[index]
    def __getitem__(self, index):
        return vectorize(self.examples[index], self.args, self.dict,self.if_train)  # train : single_answer==True
    
    def lengths(self):
        # a list, each element is one ex's doc,ques tokens' len tuple
        # [(31,5),(52,6),...]
        return [(len(ex['document']), len(ex['question']))
                for ex in self.examples]

# ------------------------------------------------------------------------------
# PyTorch sampler returning batched of sorted lengths (by doc and question).
# ------------------------------------------------------------------------------


class SortedBatchSampler(Sampler):

    def __init__(self, lengths, batch_size, shuffle=True):
        # self.lengths; each element is one ex's doc,ques tokens' len tuple
        # [(31,5),(52,6),(51,6),...]
        self.lengths = lengths             # all ex's (l_doc,l_query), given by ReaderDataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        # self.le
        # length= [(-31,-5,0.32),(-52,-6,0.11),(-51,-6,0.41)...]    for all samples
        lengths = np.array(
            [(-l[0], -l[1], np.random.random()) for l in self.lengths],
            dtype=[('l1', np.int_), ('l2', np.int_), ('rand', np.float_)]
        )
        # sorted indices accrdding to doc_len,qus_len,random        [1,2,0] 
        indices = np.argsort(lengths, order=('l1', 'l2', 'rand'))
        # batches: [[3,5,1,7,3,6],[7,9,2,5,10,12],...], sample sorted by doc len
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:# shuffle batch's order, but each batch have similar doc len
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])   # i: each sample_index in each batch

    def __len__(self):
        # all ex
        return len(self.lengths)
  
from collections import Counter
def top_question_words(args, examples, word_dict):
    """Count and return the most common question words in provided examples."""
    word_count = Counter()
    for ex in examples:
        for w in ex['question']:
            w = Dictionary.normalize(w)  # normalize(w)
            if w in word_dict:           # must in word_dict
                word_count.update([w])   # add w
    return word_count.most_common(args.tune_partial)    # most common question word  triple  [('e', 2), ('ve', 1)]
import time    
class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total

class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def entity_groups(entities,tool):
    """Group consecutive entity tokens with the same NER tag."""
    tokens = []
    NER_tokens = []
    candiate_list=[]
    NER_candiate_list=[]
    idx = 0                            # token 位置
    while idx < len(entities):
        ner_tag = entities[idx][1]
        # Check for entity tag
        if (tool=='corenlp' and ner_tag != 'O') or (tool=='spacy' and ner_tag !=''):  # # non_entity 'O'   ''
            # Chomp the sequence
            candidate=[]
            # 位置 start 到 start+index 都是同一个entity
            while (idx < len(entities) and entities[idx][1] == ner_tag):
                candidate.append(entities[idx][0])
                idx += 1

            text=' '.join(candidate)
            candiate_list.append(text)
            tokens.append(text)

            NER_text='@_'+'_'.join(candidate)
            NER_candiate_list.append(NER_text)
            NER_tokens.append(NER_text)
            #idx+=1
        else:
            text = entities[idx][0]
            tokens.append(text)
            NER_tokens.append(text)
            idx += 1

    #new_sen=' '.join(new_sen)
    #NER_new_sen=' '.join(NER_new_sen)
    # 一个list，包含text中出现的每个entity的原文，以及该entity对应的类型 组成tuple
    return tokens,NER_tokens,list(set(candiate_list)),list(set(NER_candiate_list))

# =============================================================================
#             if final_score[ex_id] > 0:
#                 Q=Q_ids[final_index[ex_id]]
#                 if Q_scores[qid].get(Q):
#                     Q_scores[qid][Q]+=final_score[ex_id]
#                 else:
#                     Q_scores[qid][Q]=final_score[ex_id]
#                     
# =============================================================================
#a=torch.sum(Q_mask,1).squeeze(1)
#assert torch.sum(final_index>=a)==0