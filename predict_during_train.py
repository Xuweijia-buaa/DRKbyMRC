#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 20:40:12 2018

@author: xuweijia
"""
from utils import ReaderDataset,SortedBatchSampler # ,index_embedding_words
from Vectorize import batchify
import math
import torch
import subprocess
import multiprocessing

def get_test_loader(test_exs,args,word_dict,feature_dict,cuda):
    examples = []    # all write like train ex, each doc with it's query
    train_mode=args.train_mode
    batch_size=args.test_batch_size
#    if use_embedding:
#        words = index_embedding_words(embedding_file)  # all words in embedding_file
#        print('words in embed:{}'.format(len(words)))
#        added =model.expand_dictionary(words)          # new words, in embedding_words but not in current word_dict
#        model.load_embeddings(added, embedding_file)
        
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
    dataset = ReaderDataset(examples, args, word_dict, feature_dict, if_train=False)   # vectorize each ex
    sampler = SortedBatchSampler(dataset.lengths(),batch_size,shuffle=False)
    # num_loaders = min(5, math.floor(len(examples) / 1e3))  
    num_loaders=args.data_workers
    loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,sampler=sampler,num_workers=num_loaders,collate_fn=batchify,pin_memory=cuda)
    return examples,dataset,loader


def predict(test_exs,Q_scores):
        # predict
    exact_match=0
    exact_match3=0
    exact_match10=0
    exclude_self=0
    
    n_ans_exist=0
    for ex_id,sample in enumerate(test_exs):
        predictions=sorted(Q_scores[ex_id].items(), key=lambda item :item[1],reverse=True)
        predictions= [p[0] for p in predictions]
        
        ans_id=sample['ans_id']
        e1_id=sample['triple'][0][0]
        ans_exist=sample['ans_exist']
        
        if len(predictions)==0:
            continue
        
        prediction=predictions[0]
        correct=prediction in ans_id
        exact_match+=correct
        n_ans_exist+=ans_exist
        
        prediction=predictions[1] if prediction[0]==e1_id and len(predictions)>1 else predictions[0]
        correct=prediction in ans_id
        exclude_self+=correct
        
        correct3=len(([p for p in predictions[:3] if p in ans_id]))!=0
        #correct3=any([p for p in predictions[:3] if p in ans_id])
        exact_match3+=correct3
        
        correct10=len(([p for p in predictions[:10] if p in ans_id]))!=0
        # correct10=any([p for p in predictions[:10] if p in ans_id])
        exact_match10+=correct10  
        
        #print(ex_id)
        #print(n_ans_exist)
        
    total=len(test_exs)
    #exact_match_exist_rate = 100.0 * exact_match_exist/ total_have
    exact_match_rate = 100.0 * exact_match / total
    exact_match_rate3 = 100.0 * exact_match3 / total
    exact_match_rate10 = 100.0 * exact_match10 / total
    
    exact_match_rate_ans_exist = 100.0 * exact_match / n_ans_exist
#    exclude_self_rate=100.0 * exclude_self / total
#    print({'exact_match': exact_match},{'total:':total}) 
#    print({'exact_match_rate': exact_match_rate})  
#    print({'exclude_self_rate': exclude_self_rate})  
#    print({'exact_match_rate3': exact_match_rate3})  
#    print({'exact_match_rate10': exact_match_rate10})  
    
    return exact_match_rate,exact_match_rate3,exact_match_rate10,exact_match_rate_ans_exist,n_ans_exist

import numpy as np
import random
def get_test_result(test_exs,model,args,loader):       
    #pool= multiprocessing.Pool(processes=4)
    pool= multiprocessing.Pool(processes=4) if args.pool else None
    Q_scores_single = [{} for _ in range(len(test_exs))]       # q_predict[q_id][c]  : q's all preidction c's  total score dict
    Q_scores_multi = [{} for _ in range(len(test_exs))]       # q_predict[q_id][c]  : q's all preidction c's  total score dict
    for idx,ex in enumerate(loader): # ex
        # B,1, each ex's predict Q's index and corresponding score = model.predict(batch, top_n=test_args.doc_top_n,async_pool=pool) 
        # final_score,final_Q_index      B,1
        final_score,final_index,Q_mask,ids = model.predict(ex, top_n=1,pool= pool, normalize_ss=False,exp_final_Q=False) 
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
            #Q_names=sample['Q_name'][d_idx]
            l_Q=np.sum(Q_mask[ex_id].numpy())             # real Q number
            assert l_Q==len(Q_ids)
            for i in range(l_Q):
                # multi
                if final_score[ex_id][i] > 0:
                    Q=Q_ids[final_index[ex_id][i]]    #  for one ex, all same Q's score sum   
                    #Q_name=Q_names[final_index[ex_id][i]]
                    if Q_scores_multi[qid].get(Q):
                        Q_scores_multi[qid][Q]+=final_score[ex_id][i]
                    else:
                        Q_scores_multi[qid][Q]=final_score[ex_id][i]
            if args.test_doc_top_1==True:   # each doc preedict 1 Q
                l_Q=1
            for i in range(l_Q):
                # single
                if final_score[ex_id][i] > 0:
                    Q=Q_ids[final_index[ex_id][i]]    #  for one ex, all same Q's score sum   
                    #Q_name=Q_names[final_index[ex_id][i]]
                    if Q_scores_single[qid].get(Q):
                        Q_scores_single[qid][Q]+=final_score[ex_id][i]
                    else:
                        Q_scores_single[qid][Q]=final_score[ex_id][i]
                        
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
            
    ls=[len(Q_dict) for Q_dict in  Q_scores_single]  # each sample have how many Q as candidates
    avg_Q=np.mean(ls)

    return Q_scores_multi,Q_scores_single,Q_scores_most,Q_scores_random,avg_Q
    
import json
def predict_rerank(args,test_exs,Q_scores,rerank_info,if_compare=False):
    transE=False
    transH=False
    transR=False
    transD=False
    if args.transX=='transE':
        transE=True
    if args.transX=='transH':
        transH=True
    if args.transX=='transR':
        transR=True
    if args.transX=='transD':
        transD=True
    filename=rerank_info['e_embed']
    p_filename=rerank_info['r_embed']
    A_filename=rerank_info['A_embed'] if args.transX!='transE' else None
    N_entity=rerank_info['E']
    N_relation=rerank_info['R']
    h=args.size
    eid2idx=rerank_info['eid2idx']
    pid2idx=rerank_info['pid2idx']
    if transE:
        entity_embedding = np.memmap(filename ,dtype='float32', shape=(N_entity,h),mode='r')
        relation_embedding = np.memmap(p_filename ,dtype='float32', shape=(N_relation,h),mode='r')
    elif transH:
        def transferH(e,r_p):
            return e - np.sum(e * r_p) * r_p
        import numpy as np
        entity_embedding = np.memmap(filename ,dtype='float32', shape=(N_entity,h),mode='r')
        relation_embedding = np.memmap(p_filename ,dtype='float32', shape=(N_relation,h),mode='r')
        relation_transfer=np.memmap(A_filename ,dtype='float32', shape=(N_relation,h),mode='r')
    elif transD:
        def transferD(e,e_p,r_p):# h,hp,rp
            return e+np.sum(e*e_p)*r_p
        entity_embedding = np.memmap(filename ,dtype='float32', shape=(N_entity,h),mode='r')
        relation_embedding = np.memmap(p_filename ,dtype='float32', shape=(N_relation,h),mode='r')
        A_embedding=np.memmap(A_filename ,dtype='float32', shape=(N_entity+N_relation,h),mode='r')
        relation_transfer=A_embedding[:N_relation]
        entity_transfer=A_embedding[N_relation:]
        
    # predict, reranking
    exact_match=0
    exact_match3=0
    exact_match10=0
    exclude_self=0
    n_ans_exist=0
    new_samples=[]
    for ex_id,sample in enumerate(test_exs):
        ans_id=sample['ans_id']
        e1_id=sample['triple'][0][0]
        p_id=sample['triple'][0][1]
        ans_exist=sample['ans_exist']
        
        predictions=sorted(Q_scores[ex_id].items(), key=lambda item :item[1],reverse=True) # Q_scores[ex_id]: Q_id:score   if compare  Qid,Qname:score
        if if_compare:
            predictions=[p[0][0] for p in predictions]  
            predictions_names=[p[0][1] for p in predictions] 
            old_pre=(predictions,predictions_names)
        else:
            predictions=[p[0] for p in predictions] 
        
        if len(predictions)==0:
            continue
        # 1
        if args.rerank_method=='soft':
            total_score=0
            my_score=[]
            for c_idx,c in enumerate(predictions):
                score=Q_scores[ex_id][(c,predictions_names[c_idx])] if if_compare else Q_scores[ex_id][c]
                total_score+=score
                my_score.append((c,score))
            my_score= dict([(c,score/total_score) for c,score in my_score]) if args.rerank_softnormal else dict(my_score)
            
        if eid2idx.get(e1_id)!=None and pid2idx.get(p_id)!=None:
            e1_idx=eid2idx[e1_id]
            p_idx=pid2idx[p_id]
            pre_indexs=[eid2idx[Q] for Q in predictions]
    # compute final score
            if transE:
                final_scores=  np.sum(abs(entity_embedding[e1_idx]+relation_embedding[p_idx]-entity_embedding[pre_indexs]),1)
            elif transH:
                p_norm=relation_transfer[p_idx]
                e1_vec=transferH(entity_embedding[e1_idx],p_norm)
                p_vec=relation_embedding[p_idx]
                final_scores=[]
                for index in pre_indexs:
                    e2_vec=transferH(entity_embedding[index],p_norm)
                    score=np.sum(abs(e1_vec+p_vec-e2_vec))
                    final_scores.append(score)
            elif transD:
                e1_vec=transferD(entity_embedding[e1_idx],entity_transfer[e1_idx],relation_transfer[p_idx])
                p_vec=relation_embedding[p_idx]
                final_scores=[]
                for index in pre_indexs:
                    e2_vec=transferD(entity_embedding[index],entity_transfer[index],relation_transfer[p_idx])
                    score=np.sum(abs(e1_vec+p_vec-e2_vec))
                    final_scores.append(score)
            new_index=np.argsort(final_scores)
            
            # 2
            if args.rerank_method=='hard':
                predictions=list(np.array(predictions)[new_index])  
            else:
                trans_score=[]
                total_score=0
                for c_idx,c in enumerate(predictions):
                    score=final_scores[c_idx]
                    total_score+=score
                    trans_score.append((c,score))
                trans_score= dict([(c,score/total_score) for c,score in trans_score]) if args.rerank_softnormal else dict(trans_score)
                real_scores=[]
                for c in predictions:
                    real_scores.append(my_score[c] -trans_score[c])
                new_index=np.argsort(-real_scores)
                predictions=list(np.array(predictions)[new_index]) 
                
            prediction=predictions[0]
            correct=prediction in ans_id
            exact_match+=correct
            n_ans_exist+=ans_exist
            
            prediction=predictions[1] if (prediction[0]==e1_id and len(predictions)>1) else predictions[0]
            correct_self=prediction in ans_id
            exclude_self+=correct_self
            
            correct3=len(([p for p in predictions[:3] if p in ans_id]))!=0
            exact_match3+=correct3
            
            correct10=len(([p for p in predictions[:10] if p in ans_id]))!=0
            exact_match10+=correct10  
            
            # sample['origin_pre'],sample['rerank_pre'],
            # sample['predict_true'],sample['rerank_true']
            # sample['state'],sample['sample_id']
            if if_compare:
                predictions_names=list(np.array(predictions_names)[new_index])
                sample['rerank_pre']=(predictions,predictions_names)
                sample['old_pre']=old_pre
                sample['ans_exist']=ans_exist
                
                old_prediction=sample['old_pre'][0][0]
                correct_old=old_prediction in ans_id
                sample['predict_true']=correct_old   
                sample['rerank_true']=correct
                
                if not sample['predict_true'] and sample['rerank_true']:
                    sample['state']= 'rerank_useful'
                    
                if sample['predict_true'] and sample['rerank_true']:
                    sample['state']= 'all_true'               
                
                if not sample['predict_true'] and not sample['rerank_true']:
                    sample['state']= 'all_false'
                    
                if sample['predict_true'] and not sample['rerank_true']:
                    sample['state']= 'rerank_wrong'
                
                sample['sample_id']=ex_id
                new_samples.append(sample)                    
        else:
            raise ValueError
    total=len(test_exs)
    #exact_match_exist_rate = 100.0 * exact_match_exist/ total_have
    exact_match_rate = 100.0 * exact_match / total
    exact_match_rate3 = 100.0 * exact_match3 / total
    exact_match_rate10 = 100.0 * exact_match10 / total
    exact_match_rate_ans_exist = 100.0 * exact_match / n_ans_exist 

    file='rerank_files'
    subprocess('mkdir','p',file)    
        
    write_file=file+'/{}_{}_{}_{}_rerank.json'.format(args.mode,args.train_mode,args.transX,args.size)
    with open(write_file,'w') as f:
        json.dump(new_samples,f)
    
    return exact_match_rate,exact_match_rate3,exact_match_rate10,exact_match_rate_ans_exist,n_ans_exist        
        
        
        
        
        

        
        
        
    