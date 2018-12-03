#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 19:37:58 2018

@author: xuweijia
"""

from collections import Counter
import torch
import numpy as np
# import torch.nn.functional as F
# span/NER test, if one doc len(ex['raw_can'][d_id])==0, 
# contain    can't find can no can find,drop

def vectorize(ex, args, dicts, if_train=True):
    """Torchify a single example."""
    word_dict = dicts[0]
    feature_dict = dicts[1] # f2fpos ,feature_dict['in_question']=0, ['in_question_uncased']=1,['in_question_lemma']=2,['pos=NN']=3,['pos=IN']=4,['pos=DT']=5,['ner=ORG']=6,...
    document = torch.LongTensor([word_dict[w] for w in ex['document']])
    question = torch.LongTensor([word_dict[w] for w in ex['question']])
    
    if if_train:
        ex['ans_in_can']=ex['E_info']['ans_in_can']
    
    if args.train_mode=='span' or args.train_mode=='string_match_base_dis':
        can=ex['can_span']                                  # ex's all can's all token spans     
    elif args.train_mode=='contain':
        can=ex['can_span']
#        can=[]
#        for c in ex['can']:
#            can.append(([word_dict[t] for t in c])) 
            
    elif args.train_mode=='NER':                          
        can=ex['can_pos']                                   # all can's all pos in doc
    elif args.train_mode=='string_match':
        can=None
        
    Q=[]  # Q[Q-dxd]: this ex's Q's all tokens
    if args.Q_mode=='Q_desp':
        for c in ex['Q_desp']:
            Q.append(torch.LongTensor([word_dict[t] for t in c]))
    elif args.Q_mode=='Q_name':
        for c in ex['Q_name']:
            Q.append(torch.LongTensor([word_dict[t] for t in c]))
    elif args.Q_mode=='Q_all':
        for qid, Q_name in enumerate(ex['Q_name']):
            Q_desp=ex['Q_desp'][qid]
            Q_all=Q_name+[':']+Q_desp
            Q.append(torch.LongTensor([word_dict[t] for t in Q_all]))
        # Q=torch.LongTensor([word_dict[t] for c in ex['Q_name'] for t in c])
        
    #                                                                  t in q /  pos/   ner /
    # D,n_feature . each token's feature vector,['I','love','you']--> [[0,1,0,/0,0,1/ 1,0,0 ]]
    if len(feature_dict) > 0:
        features = torch.zeros(len(ex['document']), len(feature_dict))
    else:
        features = None
        
    # token in question features
    if args.use_in_question:
        q_words_cased = {w for w in ex['question']}                      # qus token set
        q_words_uncased = {w.lower() for w in ex['question']}            # qus lower token set
        q_lemma = {w for w in ex['qlemma']} if args.use_lemma else None  # qus lemma token set
        for i in range(len(ex['document'])):
            # each token in doc
            if ex['document'][i] in q_words_cased:
                features[i][feature_dict['in_question']] = 1.0
            if ex['document'][i].lower() in q_words_uncased:
                features[i][feature_dict['in_question_uncased']] = 1.0
            if q_lemma and ex['lemma'][i] in q_lemma:
                features[i][feature_dict['in_question_lemma']] = 1.0
                
    # pos feature  may be 3.  001/010/001
    if args.use_pos:
        for i, w in enumerate(ex['pos']):
            f = 'pos=%s' % w                            # each token's pos name
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0      # this token corresponding column=1
    # ner feature              001/010/001
    if args.use_ner:
        for i, w in enumerate(ex['ner']):
            f = 'ner=%s' % w                            # each token's ner name
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0      # this token corresponding column=1
                
    # token t's tf in this doc tokens
    if args.use_tf:
        # counter: all this ex's doc token's tf
        counter = Counter([w.lower() for w in ex['document']])
        l = len(ex['document'])
        for i, w in enumerate(ex['document']):
            features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l

    if args.train_mode=='string_match':
        if if_train: # only train , return torch.Long 
            assert ex['ans_in_can']!=-1
            start = torch.LongTensor(1).fill_(ex['can_span'][ex['ans_in_can']][0][0])   #  ans start token pos
            end = torch.LongTensor(1).fill_( ex['can_span'][ex['ans_in_can']][0][1])    #  ans end token pos
            return document, features, question, start,end,args.train_mode,ex['id']
        else:
            return document, features, question, args.train_mode,ex['id']

    if args.train_mode=='string_match_base_dis':
        if if_train: # only train , return torch.Long
            # ex['answers']: all ans in doc, each ans is a tuple(i,j), occupy i-j token in doc  
            assert ex['ans_in_can']!=-1
            start = torch.LongTensor(1).fill_(ex['can_span'][ex['ans_in_can']][0][0])   #  ans start token pos
            end = torch.LongTensor(1).fill_( ex['can_span'][ex['ans_in_can']][0][1])    #  ans end token pos (real, included)
            return document, features, question, can, Q,ex['can2Q'],ex['triple'],ex['Q_id'],ex['Q_name'],ex['ans_in_can'], ex['ans_in_Q'],start,end,args.train_mode,ex['id']
        else:
            return document, features, question, can, Q,ex['can2Q'],ex['triple'],ex['Q_id'],ex['Q_name'],args.train_mode,ex['id']
        
    #    ex['can2Q']       # each can's correspond pos in Q  [2,3,4],[2,4],[1,2,3]
    if if_train:
        return document, features, question, can, Q,ex['can2Q'],ex['triple'],ex['Q_id'],ex['Q_name'],ex['ans_in_can'], ex['ans_in_Q'],args.train_mode,ex['id']
    else:
        return document, features, question, can, Q,ex['can2Q'],ex['triple'],ex['Q_id'],ex['Q_name'],args.train_mode,ex['id']


def batchify(batch):
    """Gather a batch of individual examples into one batch."""
    train_mode=batch[0][-2]

    ids = [ex[-1] for ex in batch]          # all batch ex id
    docs = [ex[0] for ex in batch]          # all batch ex doc (after vec)   (Long)    
    questions = [ex[2] for ex in batch]     # all batch ex qus (after vec)   (Long)
    features = [ex[1] for ex in batch]      # all batch ex feature vector  (B,D,n_f) (Long)     each ex : D,n_feature, token's feature
    B=len(batch)
    
    # Batch documents     B,D
    max_length_D = max([d.size(0) for d in docs])
    dw = torch.LongTensor(B, max_length_D).zero_()        # B,T_d  (Long),  each 
    dw_mask = torch.ByteTensor(B, max_length_D).fill_(1)  # B,T_d  (Bytes)  real len=0, false=1 
    # Batch features      B,D,n_features (concated with B,D,h)
    if features[0] is None:
        f = None
    else:
        f = torch.zeros(B, max_length_D, features[0].size(1))   # B,D,n_f       features[0]: ex 0's feature
    for i, d in enumerate(docs):
        real_len=d.size(0)
        dw[i, :real_len].copy_(d)         # doc[i]
        dw_mask[i, :real_len].fill_(0)    # real len=0,false=1 
        if f is not None:
            f[i, :real_len].copy_(features[i])  # each ex's feature : T_d,n_f
    # Batch questions    B,Q
    max_length = max([q.size(0) for q in questions])
    qw = torch.LongTensor(B, max_length).zero_()        # B,T_d  (Long),  each 
    qw_mask = torch.ByteTensor(B, max_length).fill_(1)  # B,T_d  (Bytes)  real len=0, false=1
    for i, q in enumerate(questions):
        real_len=q.size(0)
        qw[i, :real_len].copy_(q)
        qw_mask[i, :real_len].fill_(0)

    if train_mode=='string_batch':
        if len(batch[0])==5:
            return dw, f, dw_mask, qw, qw_mask,ids
        # single_ans
        elif len(batch[0])==7:
            y_s = torch.cat([ex[3] for ex in batch])  # each ex's ans start token pos in thier doc 
            y_e = torch.cat([ex[4] for ex in batch])  # each ex's ans end   token pos in thier doc
            return dw, f, dw_mask, qw, qw_mask, y_s, y_e, ids

    cans= [ex[3] for ex in batch]
    Qs=[ex[4] for ex in batch]          # each is all Q's 
    can2Qs=[ex[5] for ex in batch]      #
    triples=[ex[6] for ex in batch]     # 
    Q_ids=[ex[7] for ex in batch]
    Q_names=[ex[8] for ex in batch]       
        
    # 1 get pure Q's score   B,max_Q       , with mask no normalzie/softmax   (string_match_base,is enough)
    max_Q=0                     # max Q number among each ex
    n_Q=0                       # all Q number in this batch
    max_Q_token=0               # all Q's max token number
    for ex_id in range(B):
        n_Q+=len(Qs[ex_id])
        # print(Qs[ex_id])    # should be a list,  each ex' all Q
        # print(type(Qs[ex_id]))
        l=max(len(Q_tokens) for Q_tokens in Qs[ex_id])  # Q_tokens: one Q's all token idxs,  a torchLong Tensor 
        if len(Qs[ex_id])>max_Q:
            max_Q=len(Qs[ex_id])
        if l>max_Q_token:
            max_Q_token=l   
    #  get a: Q_before sim: n_Q,h_doc: after bilstm, final represetation, h1,h2 
    #  B,h_doc,1: doc's final represetation
    Qw = torch.LongTensor(n_Q, max_Q_token).zero_()         # n_Q,Q_d  (Long)    after Bilstm,  like query, get final repretation:  n_Q,h
    Qw_mask = torch.ByteTensor(n_Q, max_Q_token).fill_(1)
    s=0
    ex2Q=[]                                                 # ex2Q[ex_id]: all Q's pos range in all Q
    
    #Q_mask=torch.ByteTensor(B,max_Q).fill_(0)
    Q_mask=np.zeros([B,max_Q],dtype='int64')                # mask real Q. before normalize over Q     (just normalize in string_base, or not normalize)
    for ex_id in range(B):
        ex2Q.append([s,s])
        # Q_mask[ex_id,:len(Qs[ex_id])].fill_(1)
        Q_mask[ex_id,:len(Qs[ex_id])]=1
        for Q_tokens in Qs[ex_id]:
            Qw[s, :len(Q_tokens)].copy_(Q_tokens)         
            Qw_mask[s, :len(Q_tokens)].fill_(0) 
            s+=1
        ex2Q[-1][1]=s
    assert s==n_Q
    # Qw --> n_Q,Q_d,h1 -->(final state) n_Q,h1 -->(W:h1,h2) --> a: n_Q,h  --> turn to transQ --> B,max_Q,h --> *d --> final_Q : B,max_Q (pure Q)
    #    trans_Q=torch.zeros([B,max_Q,h])   # B,max_Q,h     *   doc B,h,1  -->   B,max_Q, with mask  
    #    for ex_id in range(B):
    #        start,end=ex2Q[ex_id]          # Q's pos range in all Q
    #        trans_Q[ex_id,:end-start,:]=a[start:end,:]

    # 2 mention score: B,max_num_c, with mask ( B,max_num_c)  )  (no softmax,but normalize)    just need token's s return (B,D)
    if train_mode=='contain' or train_mode=='NER':# ex's all can's all tokens/ ex all can's all pos in doc
        max_num_can= max(len(ex[3]) for ex in batch)
        C_pos=np.zeros([B,max_length_D,max_num_can],dtype='int64')    # B,D,max_c     ( s: B,1,D * after softmax,C_doc_mask)  bmm  -->B,max_c
        C_doc_mask=np.zeros([B,max_length_D],dtype='int64')           # B,D          just mask none_compute s in doc,like GA. use just in s after s's softmax
        C_mask=np.zeros([B,max_num_can],dtype='int64')                # B,max_c      after get c score, softmax on real c
#        for ex_id,can in enumerate(cans):
#            C_mask[ex_id,:len(can)]=1
#            for can_idx, can_tokens in enumerate(can):
#                if train_mode=='contain':
#                    pos= [i for i in range(len(docs[ex_id])) if docs[ex_id][i] in can_tokens]  #  each can 's all token's pos in doc
#                else:
#                    pos= can_tokens #  each replaced can's pos in doc
#                C_pos[ex_id,pos,can_idx]=1
#                C_doc_mask[ex_id,pos]=1                   
        for ex_id,can in enumerate(cans): # this ex's all span number  [2,3],   [[4,5],[5,6]],    [7,8]   4 sapns
            C_mask[ex_id,:len(can)]=1
            for can_idx,can_spans in enumerate(can): # can_idx's  can's all span 
                if train_mode=='contain':
                    pos=set()                        # each can's all token's pos
                    for can_span in can_spans:       # each can's each span  [2,2]   [3,5](include 5)  [6,6]
                        start,end=can_span
                        if start==end:
                            pos.add(start)
                        else:
                            pos|=set(range(start,end+1))
                    pos=list(pos)
                    assert max(pos)<len(docs[ex_id])
                else:
                    pos= can_spans #  each replaced can's pos in doc
                C_pos[ex_id,pos,can_idx]=1
                C_doc_mask[ex_id,pos]=1   

    # 3 get final B,max_Q,besed on mention
        # B,max_c,max_Q , each c's P(Q|c). may contain nan
        CQ_mask=np.zeros([B,max_num_can,max_Q],dtype='int64')     # B,max_c,max_Q,  to mask expanded B,max_Q, then softmax to get each P(Q|c)
        #torch.ByteTensor(n_Q, max_Q_token).fill_(1)
        #CQ_mask=torch.ByteTensor(B,max_num_can,max_Q).fill_(1)
        for ex_id in range(B):
            c2Q=can2Qs[ex_id]
            for can_id,pos in enumerate(c2Q):
                CQ_mask[ex_id,can_id,pos]=1                
        if len(batch[0])==11:
            return dw, f, dw_mask, qw, qw_mask,Qw,Qw_mask,Q_mask,C_pos,C_doc_mask,C_mask,ex2Q,CQ_mask,Q_ids,Q_names,triples,ids
        elif len(batch[0])==13:
            ans_in_can=[ex[9] for ex in batch]
            ans_in_Q=[ex[10] for ex in batch]
            return dw, f, dw_mask, qw, qw_mask,Qw,Qw_mask,Q_mask,C_pos,C_doc_mask,C_mask,ex2Q,CQ_mask,ans_in_can,ans_in_Q,Q_ids,Q_names,triples,ids
        # C_pos,C_doc_mask,C_mask,Q_mask,CQ_mask:np   ex2Q,ans_in_Q ,Q_ids(each ex's all Q), list       
                
    if train_mode=='span': # ex's all can's all token spans   need start_s,end_s return
        # 2 mention score: B,max_num_c, with mask ( B,max_num_c)  )  (no softmax,but normalize)
        max_num_can= max(len(ex[3]) for ex in batch)
        max_span=0
        for ex_id,can in enumerate(cans):
            l=0                   # this ex's all span number  [2,3],   [[4,5],[5,6]],    [7,8]   4 sapns
            for can_spans in can: # each can's all span
                l+=len(can_spans)
            if l>max_span:
                max_span=l
        start_indexs=np.zeros([B,max_span],dtype='int64')    # s_start.gather(dim=1,index=satrt_indexs)  B,D--> B,max_span * span_mask (softmax), softmax_over_span.each span's score 
        end_indexs=np.zeros([B,max_span],dtype='int64')      # s_end.gather(dim=1,index=end_indexs)      B,D--> B,max_span
        span_mask=np.zeros([B,max_span],dtype='int64') 
        span2c=np.zeros([B,max_span,max_num_can],dtype='int64')   # B,max_span,max_can  * B,1,max_span   bmm-->     B,max_num_c
        C_mask=np.zeros([B,max_num_can],dtype='int64')            # mask to B,max_num_c, normalize/softmax to mention/
        for ex_id,can in enumerate(cans):
            l=0
            starts=[span[0] for c in can for span in c if len(c)!=0]
            ends=[span[1] for c in can for span in c if len(c)!=0]
            start_indexs[ex_id,:len(starts)]=starts
            end_indexs[ex_id,:len(starts)]=ends
            span_mask[ex_id,:len(starts)]=1
            for can_idx,can_spans in enumerate(can): # each can's all span : [[1,3],[5,7]] / [[4,7]] / []
                if len(can_spans)==0:
                    continue
                span2c[ex_id,l:l+len(can_spans),can_idx]=1
                l+=len(can_spans)
        # 3 get final B,max_Q,besed on mention
        # B,max_c,max_Q , each c's P(Q|c). may contain nan
        CQ_mask=np.zeros([B,max_num_can,max_Q],dtype='int64')     # B,max_c,max_Q,  to mask expanded B,max_Q, then softmax to get each P(Q|c)
        # CQ_mask=torch.ByteTensor(B,max_num_can,max_Q).fill_(1)
        # CQ_mask=torch.ByteTensor(B,max_num_can,max_Q).fill_(0)
        for ex_id in range(B):
            c2Q=can2Qs[ex_id]
            for can_id,pos in enumerate(c2Q):
                CQ_mask[ex_id,can_id,pos]=1
                #pos=torch.LongTensor(pos)
                #CQ_mask[ex_id,can_id,pos].fill_(1) 
        if len(batch[0])==11:
            return dw, f, dw_mask, qw, qw_mask,Qw,Qw_mask,Q_mask,start_indexs,end_indexs,span_mask,span2c,C_mask,ex2Q,CQ_mask,Q_ids,Q_names,triples,ids
        elif len(batch[0])==13:
            ans_in_can=[ex[9] for ex in batch]
            ans_in_Q=[ex[10] for ex in batch]
            #print(dw, f, dw_mask, qw, qw_mask,Qw,Qw_mask,start_indexs,end_indexs,span_mask,span2c,C_mask,Q_mask,ex2Q,CQ_mask,ans_in_can,ans_in_Q,Q_ids,Q_names,triples,ids)
            return dw, f, dw_mask, qw, qw_mask,Qw,Qw_mask,Q_mask,start_indexs,end_indexs,span_mask,span2c,C_mask,ex2Q,CQ_mask,ans_in_can,ans_in_Q,Q_ids,Q_names,triples,ids  
         #  start_indexs,end_indexs,span_mask,span2c,C_mask, Q_mask, CQ_mask: np                  ex2Q,ans_in_Q: list

    if train_mode=='string_match_base_dis':
        max_num_can= max(len(ex[3]) for ex in batch)
        CQ_mask=np.zeros([B,max_num_can,max_Q],dtype='int64')     # B,max_c,max_Q,  to mask expanded B,max_Q, then softmax to get each P(Q|c)
        # CQ_mask=torch.ByteTensor(B,max_num_can,max_Q).fill_(1)
        # CQ_mask=torch.ByteTensor(B,max_num_can,max_Q).fill_(0)
        for ex_id in range(B):
            c2Q=can2Qs[ex_id]
            for can_id,pos in enumerate(c2Q):
                CQ_mask[ex_id,can_id,pos]=1
        if len(batch[0])==11:
            return dw, f, dw_mask, qw, qw_mask,Qw,Qw_mask,Q_mask,ex2Q,CQ_mask,Q_ids,Q_names,triples,cans,ids
        # single_ans
        elif len(batch[0])==15:
            y_s = torch.cat([ex[-4] for ex in batch])  # each ex's ans start token pos in thier doc 
            y_e = torch.cat([ex[-3] for ex in batch])  # each ex's ans end   token pos in thier doc
            ans_in_Q=[ex[10] for ex in batch]
            ans_in_can=[ex[9] for ex in batch]
            return dw, f, dw_mask, qw, qw_mask, Qw,Qw_mask,Q_mask,ex2Q,CQ_mask,ans_in_can,ans_in_Q,y_s, y_e,Q_ids,Q_names,triples,ids
            #Q_mask:np      ex2Q,ans_in_Q:list
            #answear=torch.Tensor(ans_in_Q)
            #answear_index=answear.unsqueeze(1) # B,1
            #predict_prob=final_Q.gather(1,answear_index.long()) # B,1 

#        CQ_mask=torch.from_numpy(CQ_mask)
#        # 1 softmax
#        CQ_mask[CQ_mask==0]=-float('inf')
#        B_max_Q=np.random.rand(B,max_Q)    # pure Q                                        
#        B_max_Q=torch.from_numpy(B_max_Q).view(B,1,max_Q)
#        B_max_Q=B_max_Q.expand_as(CQ_mask) # B,max_Q    --> B,max_c,max_Q  (give all Q' original score to c)
#        B_max_Q=B_max_Q.float()*CQ_mask.float()       # B,max_c,max_Q, after normalize, get each c's P(Q|c)            
#        B_max_Q=B_max_Q.view(-1,max_Q)
#        B_max_Q=F.softmax(Variable((B_max_Q))).view(B,-1,max_Q)
#         
         #  2 normalize  pure Q   B,max_c,max_Q/
#        Sum=torch.sum(B_max_Q,1).expand_as(B_max_Q)
#        Sum[Sum==0]=1
#        B_max_Q=(B_max_Q/Sum).view(B,-1,max_Q)          
         # 2 fisrt softmax, then mask
         # B_max_Q=F.softmax(B_max_Q, dim=-1)
#        # get final Q
#        # B,max_C (mask, normalized)--> bmm( B,1,max_c  P(c),  B,max_c,max_Q P(Q|c) ) --> B,max_Q  real P(Q) for each Q  P(Q1)=P(Q1|c1)*P(c1)+P(Q1|c3)*P(c3)
#        # final_Q=torch.bmm(B_max_c.view(B,1,max_c),B_max_Q).squeeze(1)    # B,max_Q 
         
         
#        # train
#        answear=torch.Tensor(ans_in_Q)
#        answear_index=answear.unsqueeze(1) # B,1
#        predict_prob=final_Q.gather(1,answear_index.long()) # B,1
#        loss=torch.mean(-torch.log(predict_prob))           # span / NER / contain
#        # test
#        predict_Q_scores, predict_Q_idx=torch.sort(final_Q,1,descending=True)
#        predict_Q_score, predict_Q_idx=predict_Q_score.numpy(), predict_Q_idx.numpy()
#        for ex_id in range(B):
#            predict_Q=np.array(Q_ids[ex_id])[predict_Q_idx[ex_id]]) # each ex's predict Q      ['ew', 'cxwn', 'cne', 'cnjewo', 'cnewjke']
#            predict_Q_score=predict_Q_scores[ex_id]                # and corresponding score [ 0.42593992,  0.35299581,  0.17785761,  0.04320664,  0]        
    
     
             
        
        
                
                
            
        
                
        
        
        
        