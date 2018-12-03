#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 20:40:12 2018

@author: xuweijia
"""
from utils import ReaderDataset,SortedBatchSampler,index_embedding_words
from Vectorize import batchify
from Reader import DocReader
import math
import torch
import multiprocessing
import random
import numpy as np
from predict_during_train import predict,predict_rerank
def raw_predict(test_exs,test_args,model_file,use_embedding,embedding_file,cuda,normalize_q,normalize_s,normalize_ss,rerank_info=None):
    examples = []    # all write like train ex, each doc with it's query
    model = DocReader.load(model_file, normalize_q=normalize_q,normalize_s=normalize_s)
    assert test_args.train_mode==model.args.train_mode
    train_mode=model.args.train_mode
    if cuda:
        model.cuda()
    if use_embedding:
        words = index_embedding_words(embedding_file)  # all words in embedding_file
        print('words in embed:{}'.format(len(words)))
        added =model.expand_dictionary(words)          # new words, in embedding_words but not in current word_dict
        model.load_embeddings(added, embedding_file)
    
    # get loader
    for qid,sample in enumerate(test_exs):
        for d_idx, doc in enumerate(sample['document']):
            if train_mode=='NER':
                examples.append({
                'id': (qid,d_idx),                        # this ques id
                'question': sample['question'],           # all ques tokens
                'qlemma': sample['qlemma'],               # all ques tokens's lemma
                # all doc's
                'document': doc,                                 # all doc  tokens
                'offsets': sample['offsets'][d_idx],             # all tokens's span in doc
                'lemma': sample['lemma'][d_idx],                 # all doc  tokens's lemma
                'pos': sample['pos'][d_idx],                     # all doc  tokens's pos
                'ner': sample['ner'][d_idx],                     # all doc  tokens's ner
                'raw_can': sample['raw_can'][d_idx],             # n_can: all valid cands
                'can_pos':sample['pos_c'][d_idx],                  # n_doc,n_c,max_can_pos:  all raw cands pos in doc_tokens              
                'Q_id':sample['Q_id'][d_idx],                    # each Q's id
                'Q_name':sample['Q_name'][d_idx],                # each Q's name (splited)
                'Q_desp':sample['Q_desp'][d_idx],                # each Q's desp (splited)
                'can2Q':sample['can2Q'][d_idx],                  # each can's correspond Q's pos in Q_id/Q_name/Q_desp  [2,3,4],[2,4],[1,2,3]
                'triple':sample['triple'],
                })
            else:
                examples.append({
                'id': (qid,d_idx),                        # this ques id
                'question': sample['question'],           # all ques tokens
                'qlemma': sample['qlemma'],               # all ques tokens's lemma
                # all doc's
                'document': doc,                                 # all doc  tokens
                'offsets': sample['offsets'][d_idx],             # all tokens's span in doc
                'lemma': sample['lemma'][d_idx],                 # all doc  tokens's lemma
                'pos': sample['pos'][d_idx],                     # all doc  tokens's pos
                'can':sample['can'][d_idx],                      # n_can,max_can_tokens:  all split valid cands, mask doc, to compute each mention's score              (method 2, like GA)
                'can_span':sample['can_span'][d_idx],             # n_can,max_can_show,2:   all can's all token's span, used to compute each mention score by start * end (method 1) 
                'ner': sample['ner'][d_idx],                     # all doc  tokens's ner
                'raw_can': sample['raw_can'][d_idx],             # n_can: all valid cands             
                'Q_id':sample['Q_id'][d_idx],                    # each Q's id
                'Q_name':sample['Q_name'][d_idx],                # each Q's name (splited)
                'Q_desp':sample['Q_desp'][d_idx],                # each Q's desp (splited)
                'can2Q':sample['can2Q'][d_idx],                  # each can's correspond Q's pos in Q_id/Q_name/Q_desp  [2,3,4],[2,4],[1,2,3]
                'triple':sample['triple'],
                })      

    # 输入的是被tokenize 过的 ex 写入loader
    dataset = ReaderDataset(examples, model.args, model.word_dict, model.feature_dict, if_train=False)   # vectorize each ex
    sampler = SortedBatchSampler(dataset.lengths(),test_args.batch_size,shuffle=False)
    #num_loaders = min(5, math.floor(len(examples) / 1e3))
    num_loaders=test_args.data_workers
    loader = torch.utils.data.DataLoader(dataset,batch_size=test_args.batch_size,sampler=sampler,num_workers=num_loaders,collate_fn=batchify,pin_memory=cuda)
    
    
    model.set_normalize(normalize_q,normalize_s)
    
    # get results
    pool= multiprocessing.Pool(processes=4) if test_args.pool else None
    Q_scores_single = [{} for _ in range(len(test_exs))]       # q_predict[q_id][c]  : q's all preidction c's  total score dict
    Q_scores_multi = [{} for _ in range(len(test_exs))]        # q_predict[q_id][c]  : q's all preidction c's  total score dict
    Q_scores_single_name = [{} for _ in range(len(test_exs))]       # q_predict[q_id][c]  : q's all preidction c's  total score dict
    Q_scores_multi_name = [{} for _ in range(len(test_exs))]        # q_predict[q_id][c]  : q's all preidction c's  total score dict
    for idx,ex in enumerate(loader): # ex
        # B,1, each ex's predict Q's index and corresponding score = model.predict(batch, top_n=test_args.doc_top_n,async_pool=pool) 
        # final_score,final_Q_index      B,1
        final_score,final_index,Q_mask,ids = model.predict(ex, top_n=1,pool= pool, normalize_ss=normalize_ss,exp_final_Q=False) 
        final_score, final_index,Q_mask = final_score.data.cpu(),final_index.data.cpu(),Q_mask.data.cpu()    # B,max_Q
        #final_score1,final_index1=final_score_m[:,0],final_index_m[:,0]
        #assert torch.sum(final_index1!=final_index_s)==0
        #assert torch.sum(final_score1!=final_score_s)==0
        B=final_score.size(0)
        #final_score,final_index,Q_mask = final_score.data.cpu().numpy(),final_index.data.cpu().numpy(),Q_mask.data.cpu().numpy()    # B,max_Q
        # final_score,final_index,Q_mask = final_score.data.cpu(),final_index.data.cpu(),Q_mask.data.cpu()    # B,max_Q
        # each ex in this batch
        for ex_id in range(B):
            qid,d_idx=ids[ex_id]
            sample=test_exs[qid]                   # each original sample
            Q_ids=sample['Q_id'][d_idx]           # original sample's real Q_ids
            Q_names=sample['Q_name'][d_idx]
            l_Q=np.sum(Q_mask[ex_id].numpy())             # real Q number
            assert l_Q==len(Q_ids)
            for i in range(l_Q):
                # multi
                if final_score[ex_id][i] > 0:
                    Q=Q_ids[final_index[ex_id][i]]    #  for one ex, all same Q's score sum   
                    Q_name=' '.join(Q_names[final_index[ex_id][i]])
                    if Q_scores_multi[qid].get(Q):
                        Q_scores_multi[qid][Q]+=final_score[ex_id][i]
                    else:
                        Q_scores_multi[qid][Q]=final_score[ex_id][i]     
                        
                    if Q_scores_multi_name[qid].get((Q,Q_name)):
                        Q_scores_multi_name[qid][(Q,Q_name)]+=final_score[ex_id][i]
                    else:
                        Q_scores_multi_name[qid][(Q,Q_name)]=final_score[ex_id][i]                   
            # test_doc_top_1=True   # each doc preedict 1 Q
            l_Q=1
            for i in range(l_Q):
                # single
                if final_score[ex_id][i] > 0:
                    Q=Q_ids[final_index[ex_id][i]]    #  for one ex, all same Q's score sum   
                    Q_name=' '.join(Q_names[final_index[ex_id][i]])
                    if Q_scores_single[qid].get(Q):
                        Q_scores_single[qid][Q]+=final_score[ex_id][i]
                    else:
                        Q_scores_single[qid][Q]=final_score[ex_id][i]     
                        
                    if Q_scores_single_name[qid].get((Q,Q_name)):
                        Q_scores_single_name[qid][(Q,Q_name)]+=final_score[ex_id][i]
                    else:
                        Q_scores_single_name[qid][(Q,Q_name)]=final_score[ex_id][i]   
                        
                        
    Q_scores_most=[{} for _ in range(len(test_exs))]
    Q_scores_random=[{} for _ in range(len(test_exs))]
    for qid,sample in enumerate(test_exs):
        Q_set=set()
        # Q_count={}
        for d_idx in range(len(sample['document'])):
            Q_ids=sample['Q_id'][d_idx]
            for Q in Q_ids:
                Q_set.add(Q)
                if Q_scores_most[qid].get(Q) != None:
                    Q_scores_most[qid][Q]+=1
                else:
                    Q_scores_most[qid][Q]=1        
        # Q_scores_most[qid]=Q_count
        Q_set=list(Q_set)
        random.shuffle(Q_set)
        for Q in Q_set:
            Q_scores_random[qid][Q]=0
            
    ls=[len(Q_dict) for Q_dict in  Q_scores_single]
    avg_Q=np.mean(ls)
                    
    if rerank_info==None:
        exact_match_rate_s,exact_match3_s,exact_match10_s,ex_ans_exist_s,n_ans_exist=predict(test_exs,Q_scores_single)    
        exact_match_rate_m,exact_match3_m,exact_match10_m,ex_ans_exist_m,_=predict(test_exs,Q_scores_multi)  
        
        exact_match_rate_most,exact_match3_most,exact_match10_most,ex_ans_exist_most,_=predict(test_exs,Q_scores_most)   
        exact_match_rate_rand,exact_match3_rand,exact_match10_rand,ex_ans_exist_most,_=predict(test_exs,Q_scores_random)
           
        exact_match= exact_match_rate_s if exact_match_rate_s>=exact_match_rate_m else exact_match_rate_m
        exact_match_ans= ex_ans_exist_s if ex_ans_exist_s>=ex_ans_exist_m else ex_ans_exist_m
        winning_method= 'single' if exact_match_rate_s>=exact_match_rate_m else 'multi'
        
        n_ans_exist/len(test_exs)
        print({'winning_method:':winning_method})
        print({'exact_match %': exact_match},{'total:':len(test_exs)})
        print({'exact_match_ans %': exact_match_ans},{'total_ans:':n_ans_exist})
        
        print({'exact_match_m %': exact_match_rate_m},{'exact_match_s %': exact_match_rate_s})
        print({'exact_match_m_ans %': ex_ans_exist_m},{'exact_match_s_ans %': ex_ans_exist_s})
        print({'exact_match_m %': exact_match_rate_m},{'exact_match_s %': exact_match_rate_s})
        print({'exact_match3_m': exact_match3_m},{'exact_match_3_s': exact_match3_s})
        print({'exact_match10_m': exact_match10_m},{'exact_match_10_s': exact_match10_s})
        print({'exact_match_most': exact_match_rate_most})  
        print({'exact_match_random': exact_match_rate_rand})  
        print({'avg_Q': avg_Q})  
        print({'ans_exist_rate:':n_ans_exist/len(test_exs)})
    else:
        if not test_args.rerank_compare:
            exact_match_rate_s,exact_match3_s,exact_match10_s,ex_ans_exist_s,n_ans_exist=predict_rerank(test_args,test_exs,Q_scores_single,rerank_info,if_compare=False)    
            exact_match_rate_m,exact_match3_m,exact_match10_m,ex_ans_exist_m,_=predict_rerank(test_args,test_exs,Q_scores_multi,rerank_info,if_compare=False) 
        else:
            exact_match_rate_s,exact_match3_s,exact_match10_s,ex_ans_exist_s,n_ans_exist=predict_rerank(test_args,test_exs,Q_scores_single_name,rerank_info,if_compare=True)    
            exact_match_rate_m,exact_match3_m,exact_match10_m,ex_ans_exist_m,_=predict_rerank(test_args,test_exs,Q_scores_multi_name,rerank_info,if_compare=True)             

        exact_match_rate_most,exact_match3_most,exact_match10_most,ex_ans_exist_most,_=predict(test_exs,Q_scores_most)   
        exact_match_rate_rand,exact_match3_rand,exact_match10_rand,ex_ans_exist_most,_=predict(test_exs,Q_scores_random)

        exact_match= exact_match_rate_s if exact_match_rate_s>=exact_match_rate_m else exact_match_rate_m
        exact_match_ans= ex_ans_exist_s if ex_ans_exist_s>=ex_ans_exist_m else ex_ans_exist_m
        winning_method= 'single' if exact_match_rate_s>=exact_match_rate_m else 'multi'
        
        n_ans_exist/len(test_exs)
        print({'winning_method:':winning_method})
        print({'exact_match %': exact_match},{'total:':len(test_exs)})
        print({'exact_match_ans %': exact_match_ans},{'total_ans:':n_ans_exist})
        
        print({'exact_match_m %': exact_match_rate_m},{'exact_match_s %': exact_match_rate_s})
        print({'exact_match_m_ans %': ex_ans_exist_m},{'exact_match_s_ans %': ex_ans_exist_s})
        print({'exact_match_m %': exact_match_rate_m},{'exact_match_s %': exact_match_rate_s})
        print({'exact_match3_m': exact_match3_m},{'exact_match_3_s': exact_match3_s})
        print({'exact_match10_m': exact_match10_m},{'exact_match_10_s': exact_match10_s})
        print({'exact_match_most': exact_match_rate_most})  
        print({'exact_match_random': exact_match_rate_rand})  
        print({'avg_Q': avg_Q})  
        print({'ans_exist_rate:':n_ans_exist/len(test_exs)})
    
    
    
    
    