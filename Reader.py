#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 19:49:33 2018

@author: xuweijia
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import logging
from utils import to_var,to_vars,isnan,to_vars_torch

from torch.autograd import Variable
from rnn_reader import RnnDocReader
from rnn_reader_Q import RnnDocReader_Q
logger = logging.getLogger(__name__)
class DocReader(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """
    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------
    def __init__(self, args, word_dict, feature_dict,
                 state_dict=None, normalize_q=True,normalize_s=True):
        self.args = args
        self.word_dict = word_dict
        self.args.vocab_size = len(word_dict)
        # feature to idx
        self.args.num_features = len(feature_dict)
        self.feature_dict = feature_dict            
        self.updates = 0
        self.use_cuda = False
        self.parallel = False
        self.normalize_q=normalize_q
        self.normalize_s=normalize_s

        # Building network. If normalize if false, scores are not normalized
        # 0-1 per paragraph (no softmax).
        if args.model_type == 'rnn':
            if args.train_mode=='string_match':
                self.network = RnnDocReader(args, normalize_s)
            else:
                self.network = RnnDocReader_Q(args, normalize_s)
        else:
            raise RuntimeError('Unsupported model: %s' % args.model_type)
        # Load old saved state ,(if use pretrain/ from checkpoint)  
        # include 'fix-embedding' and model state-dict  / just return model state-dict
        if state_dict:
            # Load buffer separately
            if 'fixed_embedding' in state_dict:
                fixed_embedding = state_dict.pop('fixed_embedding')                 # pop fix embeddings
                self.network.load_state_dict(state_dict)                            # model parameters
                self.network.register_buffer('fixed_embedding', fixed_embedding)    # name of the buffer.buffer can be accessed from this module using the given name
            else:
                self.network.load_state_dict(state_dict)    
    # output s  /  start_end
    def update(self, ex):        
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')
        # Train mode
        self.network.train()
        train_mode=self.args.train_mode
        # old
        if train_mode=='string_match':
            # Transfer to GPU
            if self.use_cuda:
                #  x1, x1_f, x1_mask, x2, x2_mask
                inputs = [e if e is None else Variable(e.cuda(async=True)) for e in ex[:5]]  # async=True参数到cuda()的调用。这可以用于将数据传输与计算重叠
                target_s = Variable(ex[5].cuda(async=True))                 # y_s: (B, )  each ex's ans start token pos in thier doc 
                target_e = Variable(ex[6].cuda(async=True))                 # y_e: (B, )  each ex's ans end token   pos in thier doc 
            else:
                inputs = [e if e is None else Variable(e) for e in ex[:5]]
                target_s = Variable(ex[5])
                target_e = Variable(ex[6])
            score_s, score_e = self.network(*inputs)  # B,T_d, after log softmax.   ques to each doc's attetion
            # Compute loss and accuracies
            # score_s: B,Td     B sample, Td classes log prob     Td=C
            # target:  B,       each sample's real class          each value in [0,C)==[0,Td)
            loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)
            
        # new   
        if train_mode=='string_match_base_dis':
            # Q_mask:np      ex2Q,ans_in_Q:list
            dw, f, dw_mask, qw, qw_mask, Qw,Qw_mask,Q_mask,ex2Q,CQ_mask,ans_in_can,ans_in_Q,y_s, y_e,Q_ids,Q_names,triples,ids=ex
            dw, f, dw_mask, qw, qw_mask, Qw,Qw_mask = to_vars_torch(ex[:7],self.use_cuda)
            Q_mask,CQ_mask=to_vars([Q_mask,CQ_mask], use_cuda=self.use_cuda)
            if self.use_cuda:
                target_s = Variable(y_s.cuda(async=True))                 # y_s: (B, )  each ex's ans start token pos in thier doc 
                target_e = Variable(y_e.cuda(async=True))                 # y_e: (B, )  each ex's ans end token   pos in thier doc 
            else:
                target_s = Variable(y_s)
                target_e = Variable(y_e)
                
            inputs=[dw, f, dw_mask, qw, qw_mask, Qw,Qw_mask,Q_mask,ex2Q]
            # return score_s, score_e, (after log softmax), Q_pure ( B,max_Q, before Q_mask,)     
            score_s, score_e ,pure_Q = self.network(*inputs)  # B,T_d, , to compute loss 1               test, just use to predict span
            
            loss1 = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)
            #print({'loss1:':loss1})
            
            # P(c|Q), consider mention
            pure_Q=pure_Q.clone()
            pure_Q[Q_mask.data==0]=-float('inf')
            #pure_Q.data.masked_fill_(Q_mask.data==0,-float('inf')) 
            B_max_Q=pure_Q.unsqueeze(1).expand_as(CQ_mask).clone()   # B,max_Q  --> B,max_c,max_Q  (give all Q' original score to c)
            # B_max_Q_old1=B_max_Q.clone()
            B_max_Q[CQ_mask.data==0]=-float('inf')                   # B,max_c,max_Q            mask, get each c's real Q
            B_max_Q=F.softmax(B_max_Q.view(-1,B_max_Q.size(2))).view(B_max_Q.size(0),-1,B_max_Q.size(2))  # B,max_c,max_Q , get each c's P(Q|c)  some max_c 's, have no Q
            # have max_can line, no nan/  other line, all nan
            # B_max_Q.data.masked_fill_(isnan(B_max_Q.data.cpu()).cuda(),1)   # log(p==1)==0, no loss
            # B_max_Q_old2=B_max_Q.clone()
            B_max_Q=B_max_Q.clone()
            B_max_Q[isnan(B_max_Q.data.cpu()).cuda()]=1
            # print(B_max_Q)
            # P(Q)
            ans_in_can=to_var(np.array(ans_in_can,dtype='int64'),use_cuda=self.args.cuda)           # B,
            ans_index=ans_in_can.unsqueeze(1).expand(B_max_Q.size(0),B_max_Q.size(2)).unsqueeze(1)  # B,1,max_Q
            final_Q=B_max_Q.gather(1,ans_index).squeeze(1)                                          # B,max_Q 
            # print(final_Q)
            assert torch.sum(isnan(final_Q.data.cpu()))==0
            # ans in Q
            answear=to_var(np.array(ans_in_Q,dtype='int64'),use_cuda=self.args.cuda)

            if self.args.db_softmax:
                loss2 =F.nll_loss(F.log_softmax(final_Q),answear)
            else:
                answear_index=answear.unsqueeze(1) # B,1
                predict_prob=final_Q.gather(1,answear_index.long()) # B,1
                loss2=torch.mean(-torch.log(predict_prob))           # loss1 span / NER / contain   # get inf: ans_in_Q may wrong  Q_pos==0  
            #print({'loss2:':loss2})
            loss=loss1+loss2
            # test score_e,score_s (exp) get candidate mention. B mentions --> B  ans_in_can,  get B_max_Q,(exp)-->(ans_in_can)--> final_Q -->  similar to others --> get B Q_predictions
                        
        if train_mode=='contain' or train_mode=='NER':
            # C_pos,C_doc_mask,C_mask,Q_mask,CQ_mask:np   ex2Q,ans_in_Q ,Q_ids(each ex's all Q), list 
            dw, f, dw_mask, qw, qw_mask,Qw,Qw_mask,Q_mask,C_pos,C_doc_mask,C_mask,ex2Q,CQ_mask,ans_in_can,ans_in_Q,Q_ids,Q_names,triples,ids=ex
            dw, f, dw_mask, qw, qw_mask, Qw,Qw_mask = to_vars_torch(ex[:7],self.use_cuda)
            C_pos,C_doc_mask,C_mask,Q_mask,CQ_mask=to_vars([C_pos,C_doc_mask,C_mask,Q_mask,CQ_mask], use_cuda=self.args.cuda)
            
            inputs=[dw, f, dw_mask, qw, qw_mask,Qw,Qw_mask,Q_mask,ex2Q]
            # return s (after doc mask + softmax ), Q_pure (B,max_Q, before Q_mask) 
            score,pure_Q=self.network(*inputs)
            # P(c)       
            s_masked=score*C_doc_mask.float()                                  # B,D    only keep candidate in s
            s_masked+=0.00001
            s_normal=s_masked/torch.sum(s_masked,dim=1).expand_as(s_masked)    # B,D    normalize s in candidate
            B_max_c=torch.bmm(s_normal.unsqueeze(1),C_pos.float()).squeeze(1)  # B,max_c   s: B,1,D *  B,D,max_c   sum c's pos in s  already c's prob,sum==1
            B_max_c=B_max_c/torch.sum(B_max_c,dim=1).expand_as(B_max_c)        # just train

            assert torch.sum(isnan(B_max_c.data.cpu()))==0 
            # P(c|Q), consider mention
            # pure_Q.data.masked_fill_(Q_mask.data==0,-float('inf'))
            pure_Q=pure_Q.clone()
            pure_Q[Q_mask.data==0]=-float('inf')
            B_max_Q=pure_Q.unsqueeze(1).expand_as(CQ_mask).clone()   # B,max_Q  --> B,max_c,max_Q  (give all Q' original score to c)
            B_max_Q[CQ_mask.data==0]=-float('inf')                  # B,max_c,max_Q            mask, get each c's real Q
            B_max_Q=F.softmax(B_max_Q.view(-1,B_max_Q.size(2))).view(B_max_Q.size(0),-1,B_max_Q.size(2))  # B,max_c,max_Q , get each c's P(Q|c)  some max_c 's, have no Q
            # have max_can line, no nan/  other line, all nan
            # B_max_Q.data.masked_fill_(isnan(B_max_Q.data.cpu()).cuda(),1)   # log(p==1)==0, no loss
            B_max_Q=B_max_Q.clone()
            B_max_Q[isnan(B_max_Q.data.cpu()).cuda()]=1
            # P(Q)
            final_Q=torch.bmm(B_max_c.unsqueeze(1),B_max_Q).squeeze(1) # B,max_Q : bmm( B,1,max_c  P(c),  B,max_c,max_Q P(Q|c))    
            assert torch.sum(isnan(final_Q.data.cpu()))==0 
            answear=to_var(np.array(ans_in_Q,dtype='int64'),use_cuda=self.args.cuda)
            if self.args.db_softmax:
                loss =F.nll_loss(F.log_softmax(final_Q),answear)
            else:
                answear_index=answear.unsqueeze(1) # B,1
                predict_prob=final_Q.gather(1,answear_index.long()) # B,1
                loss=torch.mean(-torch.log(predict_prob))           # loss1 span / NER / contain   # get inf: ans_in_Q may wrong  Q_pos==0  
            # add m_loss
            if self.args.m_loss:
                ans_in_can=to_var(np.array(ans_in_can,dtype='int64'),use_cuda=self.args.cuda)           # B,1
                if self.args.db_softmax:
                    loss=loss+F.nll_loss(F.log_softmax(B_max_c),ans_in_can)  
                else:
                    predict_prob=B_max_c.gather(1,ans_in_can.long().unsqueeze(1)) # B,1
                    loss=loss+torch.mean(-torch.log(predict_prob)) 
            
        if train_mode=='span':
            # start_indexs,end_indexs,span_mask,span2c,C_mask, Q_mask, CQ_mask: np                  ex2Q,ans_in_Q: list
            # print(ex)
            # print(type(ex))
            dw, f, dw_mask, qw, qw_mask,Qw,Qw_mask,Q_mask,start_indexs,end_indexs,span_mask,span2c,C_mask,ex2Q,CQ_mask,ans_in_can,ans_in_Q,Q_ids,Q_names,triples,ids=ex
            dw, f, dw_mask, qw, qw_mask, Qw,Qw_mask = to_vars_torch(ex[:7],self.use_cuda)
            #CQ_mask=to_vars_torch([CQ_mask],self.use_cuda)
            Q_mask,CQ_mask,start_indexs,end_indexs,span_mask,span2c,C_mask=to_vars\
            ([Q_mask,CQ_mask,start_indexs,end_indexs,span_mask,span2c,C_mask], use_cuda=self.args.cuda)
            
            inputs=[dw, f, dw_mask, qw, qw_mask,Qw,Qw_mask,Q_mask,ex2Q]
            # return score_s, score_e, (after doc mask ,softmax), Q_pure (B,max_Q, before mask) 
            
            
            score_s, score_e,pure_Q=self.network(*inputs)
            # score_s, score_e,pure_Q,ques_final,doc_output=self.network(*inputs)
            # score_s_old, score_e_old,pure_Q_old,ques_final_old,doc_output_old=score_s.clone(), score_e.clone(),pure_Q.clone(),ques_final.clone(),doc_output.clone()
            # combine start_indexs,end_indexs,span_mask,span2c,C_mask to compute B,max_c      combile Q_mask pure Q       combine CQ_mask  real Q 
            # P(c)
            span_start=score_s.gather(dim=1,index=start_indexs) #  B,D--> B,max_span * span_mask (softmax), softmax_over_span.each span's score 
            span_end=score_e.gather(dim=1,index=end_indexs)     #  B,D--> B,max_span
            span_s=span_start*span_end*span_mask.float()
            span_s+=0.00001
            span_normal=span_s/torch.sum(span_s,dim=1).expand_as(span_s)           # normalize  B,max_span, each span's score, after mask
            B_max_c=torch.bmm(span_normal.unsqueeze(1),span2c.float()).squeeze(1)  # B,1,max_span  B,max_span,max_can     bmm-->     B,max_num_c
            # B_max_c.data.masked_fill_(C_mask.data==0,-float('inf'))            
            # B_max_c=F.softmax(B_max_c.data)                                   # B,max_c, after softmax   P(c)
            # print(B_max_c)
            # span_mask_old,span_start_old,span_end_old,span_s_old,span_normal_old,B_max_c_old=span_mask.clone(),span_start.clone(),span_end.clone(),span_s.clone(),span_normal.clone(),B_max_c.clone()
            assert torch.sum(isnan(B_max_c.data.cpu()))==0         
               
            # P(c|Q), consider mention
            # pure_Q.data.masked_fill_(Q_mask.data==0,-float('inf'))
            pure_Q=pure_Q.clone()
            pure_Q[Q_mask.data==0]=-float('inf')
            B_max_Q=pure_Q.unsqueeze(1).expand_as(CQ_mask).clone()   # B,max_Q  --> B,max_c,max_Q  (give all Q' original score to c)
            B_max_Q[CQ_mask.data==0]=-float('inf')
            B_max_Q=F.softmax(B_max_Q.view(-1,B_max_Q.size(2))).view(B_max_Q.size(0),-1,B_max_Q.size(2))  # B,max_c,max_Q , get each c's P(Q|c)  some max_c 's, have no Q
            # have max_can line, no nan/  other line, all nan
            # B_max_Q.data.masked_fill_(isnan(B_max_Q.data.cpu()).cuda(),1)  
            B_max_Q=B_max_Q.clone()
            B_max_Q[isnan(B_max_Q.data.cpu()).cuda()]=1
            # pure_Q_old2,B_max_Q_old=pure_Q.clone(),B_max_Q.clone()
            # P(Q)
            final_Q=torch.bmm(B_max_c.unsqueeze(1),B_max_Q).squeeze(1) # B,max_Q : bmm( B,1,max_c  P(c),  B,max_c,max_Q P(Q|c))  
            # print(final_Q)
            assert torch.sum(isnan(final_Q.data.cpu()))==0 
            answear=to_var(np.array(ans_in_Q,dtype='int64'),use_cuda=self.args.cuda)
            if self.args.db_softmax:
                loss =F.nll_loss(F.log_softmax(final_Q),answear)
            else:
                answear_index=answear.unsqueeze(1) # B,1
                predict_prob=final_Q.gather(1,answear_index.long()) # B,1
                loss=torch.mean(-torch.log(predict_prob))           # loss1 span / NER / contain   # get inf: ans_in_Q may wrong  Q_pos==0  
            # add m_loss
            if self.args.m_loss:
                ans_in_can=to_var(np.array(ans_in_can,dtype='int64'),use_cuda=self.args.cuda)           # B,1
                if self.args.db_softmax:
                    loss=loss+F.nll_loss(F.log_softmax(B_max_c),ans_in_can)  
                else:
                    predict_prob=B_max_c.gather(1,ans_in_can.long().unsqueeze(1)) # B,1
                    loss=loss+torch.mean(-torch.log(predict_prob)) 
        # print(loss)
        # Clear gradients and run backward
        self.optimizer.zero_grad()
        # loss=torch.sum(B_max_Q[:,0,0])
        # import copy
        # embed0=copy.deepcopy(self.network.embedding)
        # network0=copy.deepcopy(self.network)
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(),self.args.grad_clipping)

        # Update parameters
        self.optimizer.step()
        self.updates += 1
        # embed1=self.network.embedding
        # network1=self.network
        
        # Reset any partially fixed parameters (e.g. rare words)(in place)
        self.reset_parameters()
        # batch_loss, batch_size  (loss,number, give cutils.ount class)
        # return score_e,score_s,target_e,target_s,final_Q,pure_Q,B_max_Q,B_max_Q_old1,B_max_Q_old2,ans_in_can,ans_index,answear,predict_prob,loss1,loss2,loss,Q_mask,CQ_mask
        #span_mask,span2c,C_mask,score_s, score_e,span_start,span_end,start_indexs,end_indexs,span_s,span_normal,pure_Q,answear
        #return final_Q,loss,answear,span2c,span_normal,span_start,span_end,span_s,span_mask,score_e,score_s,pure_Q,B_max_c,ans_in_can,Q_mask,CQ_mask,ques_final,doc_output,dw_mask,\
         #      score_s_old, score_e_old,pure_Q_old,ques_final_old,doc_output_old,span_mask_old,span_start_old,span_end_old,span_s_old,span_normal_old,B_max_c_old,pure_Q_old2,B_max_Q_old
        # return span2c,span_normal,span_start,span_end,span_s,span_mask,score_s_old, score_e_old,score_e,score_s,final_Q,pure_Q,B_max_c,B_max_Q,ans_in_can,answear,loss,Q_mask,CQ_mask,ques_final,doc_output,dw_mask
        return loss.data[0], ex[0].size(0)
        # return loss.data[0], ex[0].size(0),pure_Q,pure_Q_old2,B_max_Q,final_Q,CQ_mask,Q_mask,pure_Q_old,B_max_Q_old1,B_max_Q_old2
        # test: number, index : torch.max(final_Q,-1)  B,1     s: exp, (give candidate),find all can's score:B,max_c    still normalize  / B_max_Q  : exp
        #       scores, indexs:torch.sort(final_Q,-1)  B,max_Q    
    def predict(self,ex,top_n=1,pool=None,normalize_ss=False,exp_final_Q=False): 
        self.network.eval()
        train_mode=self.args.train_mode
        # old
        if train_mode=='string_match':
            if self.use_cuda:
                inputs = [e if e is None else Variable(e.cuda(async=True), volatile=True) for e in ex[:5]]
            else:
                inputs = [e if e is None else Variable(e, volatile=True) for e in ex[:5]]
            score_s, score_e = self.network(*inputs)  # no normalize, just exp
            # Decode predictions
            score_s = score_s.data.cpu()
            score_e = score_e.data.cpu()  
            max_len=15
            args = (score_s, score_e, top_n, max_len)
            # return 
            # pred_s :B,top_n    each ex's top_n start token pos
            # pred_e :B,top_n    each ex's top_n end   token pos
            # pred_score: B,top_n    each ex's top_n span's score
            if pool:
                return pool.apply_async(self.decode, args)
            else:
                return self.decode(*args)    
        
        if train_mode=='string_match_base_dis':
            dw, f, dw_mask, qw, qw_mask, Qw,Qw_mask,Q_mask,ex2Q,CQ_mask,Q_ids,Q_names,triples,cans,ids=ex
            dw, f, dw_mask, qw, qw_mask, Qw,Qw_mask = to_vars_torch(ex[:7],self.use_cuda,evaluate=True)
            [Q_mask,CQ_mask]=to_vars([Q_mask,CQ_mask], use_cuda=self.use_cuda,evaluate=True)   
            inputs=[dw, f, dw_mask, qw, qw_mask, Qw,Qw_mask,Q_mask,ex2Q]
            # return score_s, score_e, (after log softmax), Q_pure ( B,max_Q, before Q_mask,)     
            score_s, score_e ,pure_Q = self.network(*inputs)  # B,T_d,               test, just use to predict span
            # Decode predictions
            score_s = score_s.data.cpu()
            score_e = score_e.data.cpu()  
            max_len=15
            args = (score_s, score_e, cans, top_n, max_len)
            # cans: ex's all can's all token spans 
            if pool:
                handle=pool.apply_async(self.decode_candidates, args)
                ans_in_can,scores=handle.get()
            else:
                ans_in_can,scores=self.decode_candidates(*args)              
            # P(c|Q), consider mention
            pure_Q=pure_Q.clone()
            pure_Q[Q_mask.data==0]=-float('inf')
            #pure_Q.data.masked_fill_(Q_mask.data==0,-float('inf'))    
            
            B_max_Q=pure_Q.unsqueeze(1).expand_as(CQ_mask).clone()   # B,max_Q  --> B,max_c,max_Q  (give all Q' original score to c)
            B_max_Q[CQ_mask.data==0]=-float('inf')
            if self.normalize_q:            
                B_max_Q=F.softmax(B_max_Q.view(-1,B_max_Q.size(2))).view(B_max_Q.size(0),-1,B_max_Q.size(2))  # B,max_c,max_Q , get each c's P(Q|c)  some max_c 's, have no Q
            else:
                B_max_Q=torch.exp(B_max_Q)  
            # B_max_Q.data.masked_fill_(isnan(B_max_Q.data.cpu()).cuda(),0)  # some lines are invalid (lines >real can)
            B_max_Q=B_max_Q.clone()
            B_max_Q[isnan(B_max_Q.data.cpu()).cuda()]=0
            
            # P(Q)
            ans_in_can=to_var(np.array(ans_in_can,dtype='int64'),use_cuda=self.args.cuda)           # B,
            ans_index=ans_in_can.unsqueeze(1).expand(B_max_Q.size(0),B_max_Q.size(2)).unsqueeze(1)  # B,1,max_Q
            final_Q=B_max_Q.gather(1,ans_index).squeeze(1)  
            assert torch.sum(isnan(final_Q.data.cpu()))==0 
            #final_score,final_index=torch.max(final_Q,-1)                          # B,1
            final_score,final_index=torch.sort(final_Q,-1,descending=True)
            # return Q_mask,B_max_Q,final_Q,final_score,final_index
            return final_score,final_index,Q_mask,ids # B,1, each ex's predict Q's index and corresponding score
            # test: number, index : torch.max(final_Q,-1)  B,1     s: exp, (give candidate),find all can's score:B,max_c    still normalize  / B_max_Q  : exp
            #       scores, indexs:torch.sort(final_Q,-1)  B,max_Q           
            
        if train_mode=='contain' or train_mode=='NER':
            # C_pos,C_doc_mask,C_mask,Q_mask,CQ_mask:np   ex2Q,ans_in_Q ,Q_ids(each ex's all Q), list 
            dw, f, dw_mask, qw, qw_mask,Qw,Qw_mask,Q_mask,C_pos,C_doc_mask,C_mask,ex2Q,CQ_mask,Q_ids,Q_names,triples,ids=ex
            dw, f, dw_mask, qw, qw_mask,Qw,Qw_mask = to_vars_torch(ex[:7],self.use_cuda,evaluate=True)
            C_pos,C_doc_mask,C_mask,Q_mask,CQ_mask=to_vars([C_pos,C_doc_mask,C_mask,Q_mask,CQ_mask], use_cuda=self.use_cuda,evaluate=True)
            
            inputs=[dw, f, dw_mask, qw, qw_mask,Qw,Qw_mask,Q_mask,ex2Q]
            # return s (after doc mask + softmax ), Q_pure (B,max_Q, before Q_mask) 
            score,pure_Q=self.network(*inputs)
            # P(c)       
            s_masked=score*C_doc_mask.float()                                  # B,D    only keep candidate in s
            # keep watching
            if normalize_ss:
                s_masked+=0.00001
                s_normal=s_masked/torch.sum(s_masked,dim=1).expand_as(s_masked)# B,D    normalize s in candidate
            else:
                s_normal=s_masked
            # s_normal=s_masked/torch.sum(s_masked,dim=1).expand_as(s_masked)    # B,D    normalize s in candidate
            B_max_c=torch.bmm(s_normal.unsqueeze(1),C_pos.float()).squeeze(1)          # B,max_c   s: B,1,D *  B,D,max_c   sum c's pos in s  already c's prob,sum==1
            # B_max_c=B_max_c/torch.sum(B_max_c,dim=1).expand_as(B_max_c)        # just train
            # B_max_c.data.masked_fill_(C_mask.data==0,-float('inf'))            
            # B_max_c=F.softmax(B_max_c.data)                                    # B,max_c, after softmax   P(c)
            assert torch.sum(isnan(B_max_c.data.cpu()))==0  
            # P(c|Q), consider mention
            # pure_Q.data.masked_fill_(Q_mask.data==0,-float('inf'))
            pure_Q=pure_Q.clone()
            pure_Q[Q_mask.data==0]=-float('inf')
            B_max_Q=pure_Q.unsqueeze(1).expand_as(CQ_mask).clone()   # B,max_Q  --> B,max_c,max_Q  (give all Q' original score to c)
            B_max_Q[CQ_mask.data==0]=-float('inf')
            if self.normalize_q:            
                B_max_Q=F.softmax(B_max_Q.view(-1,B_max_Q.size(2))).view(B_max_Q.size(0),-1,B_max_Q.size(2))  # B,max_c,max_Q , get each c's P(Q|c)  some max_c 's, have no Q
            else:
                B_max_Q=torch.exp(B_max_Q)  
            B_max_Q=B_max_Q.clone()
            B_max_Q[isnan(B_max_Q.data.cpu()).cuda()]=0
            # B_max_Q.data.masked_fill_(isnan(B_max_Q.data.cpu()).cuda(),0)  # some lines are invalid (lines >real can)
            # P(Q)
            final_Q=torch.bmm(B_max_c.unsqueeze(1),B_max_Q).squeeze(1) # B,max_Q : bmm( B,1,max_c  P(c),  B,max_c,max_Q P(Q|c))
            final_Q=torch.exp(final_Q) if exp_final_Q else final_Q
            assert torch.sum(isnan(final_Q.data.cpu()))==0 
            
            # final_score,final_index=torch.max(final_Q,-1)            # B,1
            # return Q_mask,B_max_c,B_max_Q,final_Q,final_score,final_index
            final_score,final_index=torch.sort(final_Q,-1,descending=True)
            return final_score,final_index,Q_mask,ids # B,1, each ex's predict Q's index and corresponding score
            # test: number, index : torch.max(final_Q,-1)  B,1     s: exp, (give candidate),find all can's score:B,max_c    still normalize  / B_max_Q  : exp
            #       scores, indexs:torch.sort(final_Q,-1)  B,max_Q
            
        if train_mode=='span':
            # start_indexs,end_indexs,span_mask,span2c,C_mask, Q_mask, CQ_mask: np                  ex2Q,ans_in_Q: list
            dw, f, dw_mask, qw, qw_mask,Qw,Qw_mask,Q_mask,start_indexs,end_indexs,span_mask,span2c,C_mask,ex2Q,CQ_mask,Q_ids,Q_names,triples,ids=ex
            dw, f, dw_mask, qw, qw_mask,Qw,Qw_mask = to_vars_torch(ex[:7],self.use_cuda,evaluate=True)
            start_indexs,end_indexs,span_mask,span2c,C_mask, Q_mask, CQ_mask=to_vars\
            ([start_indexs,end_indexs,span_mask,span2c,C_mask, Q_mask, CQ_mask], use_cuda=self.use_cuda,evaluate=True)
            inputs=[dw, f, dw_mask, qw, qw_mask,Qw,Qw_mask,Q_mask,ex2Q]
            # return score_s, score_e, (after doc mask ,softmax), Q_pure (B,max_Q, before mask) 
            score_s, score_e,pure_Q=self.network(*inputs)
            
            # combine start_indexs,end_indexs,span_mask,span2c,C_mask to compute B,max_c      combile Q_mask pure Q       combine CQ_mask  real Q 
            # P(c)
            span_start=score_s.gather(dim=1,index=start_indexs) #  B,D--> B,max_span * span_mask (softmax), softmax_over_span.each span's score 
            span_end=score_e.gather(dim=1,index=end_indexs)     #  B,D--> B,max_span
            span_s=span_start*span_end*span_mask.float()
            if normalize_ss:
                span_s+=0.00001
                span_normal=span_s/torch.sum(span_s,dim=1).expand_as(span_s)           # normalize  B,max_span, each span's score, after mask
            else:
                span_normal=span_s
            #span_normal=span_s/torch.sum(span_s,dim=1).expand_as(span_s)
            B_max_c=torch.bmm(span_normal.unsqueeze(1),span2c.float()).squeeze(1)  # B,1,max_span  B,max_span,max_can     bmm-->     B,max_num_c
            # B_max_c.data.masked_fill_(C_mask.data==0,-float('inf'))            
            # B_max_c=F.softmax(B_max_c.data)                                   # B,max_c, after softmax   P(c)
            assert torch.sum(isnan(B_max_c.data.cpu()))==0  
                      
            # P(c|Q), consider mention
            # pure_Q.data.masked_fill_(Q_mask.data==0,-float('inf'))
            pure_Q=pure_Q.clone()
            pure_Q[Q_mask.data==0]=-float('inf')
            B_max_Q=pure_Q.unsqueeze(1).expand_as(CQ_mask).clone()   # B,max_Q  --> B,max_c,max_Q  (give all Q' original score to c)
            B_max_Q[CQ_mask.data==0]=-float('inf')
            if self.normalize_q:            
                B_max_Q=F.softmax(B_max_Q.view(-1,B_max_Q.size(2))).view(B_max_Q.size(0),-1,B_max_Q.size(2))  # B,max_c,max_Q , get each c's P(Q|c)  some max_c 's, have no Q
            else:
                B_max_Q=torch.exp(B_max_Q)  # B,max_c,max_Q , get each c's P(Q|c)  some max_c 's, have no Q
            # have max_can line, no nan/  other line, all nan
            B_max_Q=B_max_Q.clone()
            B_max_Q[isnan(B_max_Q.data.cpu()).cuda()]=0
            # B_max_Q.data.masked_fill_(isnan(B_max_Q.data.cpu()).cuda(),0)  # some lines are invalid (lines >real can)
            
            # P(Q)
            final_Q=torch.bmm(B_max_c.unsqueeze(1),B_max_Q).squeeze(1) # B,max_Q : bmm( B,1,max_c  P(c),  B,max_c,max_Q P(Q|c))
            final_Q=torch.exp(final_Q) if exp_final_Q else final_Q
            assert torch.sum(isnan(final_Q.data.cpu()))==0 
            
            final_score,final_index=torch.sort(final_Q,-1,descending=True)
            # final_score1,final_index1=final_score1[:,0],final_index1[:,0]
            
            # final_score2,final_index2=torch.max(final_Q,-1)                          # B,1
            # final_score2,final_index2=final_score2.squeeze(1),final_index2.squeeze(1)
            # print(final_Q)
            #print(final_index)
            #print(Q_mask)
            # return Q_mask,B_max_c,B_max_Q,final_Q,final_score,final_index
            return final_score,final_index,Q_mask,ids
            # return final_score,final_index,Q_mask,ids # B,1, each ex's predict Q's index and corresponding score      
            # test: number, index : torch.max(final_Q,-1)  B,1        s: exp, (give candidate),find all can's score:B,max_c    still normalize  / B_max_Q  : exp
            #       scores, indexs:torch.sort(final_Q,-1)  B,max_Q
            

    # words (in w_dict) use pre-tained embedding, get new self.network.embedding (in place change)
    def load_embeddings(self, words, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.

        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        # words that in current w_dict
        words = {w for w in words if w in self.word_dict}     
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))
        # current embedding, inplace change
        embedding = self.network.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file) as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1)
                # w in embed_file
                w = self.word_dict.normalize(parsed[0])
                if w in words:
                    # w in words use pre-tained embedding
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.word_dict[w]].copy_(vec)
                    else:
                        logging.warning(
                            'WARN: Duplicate embedding found for %s' % w
                        )
                        vec_counts[w] = vec_counts[w] + 1
                        # In-place version of add()
                        embedding[self.word_dict[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))
        
    # tuned words embeddding       2:2+args.tune
    # remain unchange:             2+args.tune: last become model.'fixed_embedding'
    def tune_embeddings(self, words):
        # tune embedding of these words, all in current dict
        """Unfix the embeddings of a list of words. This is only relevant if
        only some of the embeddings are being tuned (tune_partial = N).

        Shuffles the N specified words to the front of the dictionary, and saves
        the original vectors of the other N + 1:vocab words in a fixed buffer.

        Args:
            words: iterable of tokens contained in dictionary.
        """
        words = {w for w in words if w in self.word_dict}

        if len(words) == 0:
            logger.warning('Tried to tune embeddings, but no words given!')
            return

        if len(words) == len(self.word_dict):
            logger.warning('Tuning ALL embeddings in dictionary')
            return

        # Shuffle words and vectors
        embedding = self.network.embedding.weight.data
        # bianli these words
        for idx, swap_word in enumerate(words, self.word_dict.START):  # start from idx 2  (0,1 unk,null)
            # Get current word + embedding for this index
            curr_word = self.word_dict[idx]
            curr_emb = embedding[idx].clone()
            # word original pos
            old_idx = self.word_dict[swap_word]

            # Swap embeddings + dictionary indices
            embedding[idx].copy_(embedding[old_idx])    # put these word's  embedding into 2,3,4...place
            embedding[old_idx].copy_(curr_emb)          # orinial 2,3,4 embedding to replace these word's pos
            # thses word idx become new_idc  2,3,4,5...
            self.word_dict[swap_word] = idx
            self.word_dict[idx] = swap_word
            # original 2,3,4 words idx become these word's old idx
            self.word_dict[curr_word] = old_idx
            self.word_dict[old_idx] = curr_word

        # Save the original, fixed embeddings
        # 注册不应被视为模型参数的缓冲区 .no model parameter,no grad
        self.network.register_buffer(
            'fixed_embedding', embedding[idx + 1:].clone()
        )
        
        # self.optimizer
    def init_optimizer(self, state_dict=None):
        """Initialize an optimizer for the free parameters of the network.

        Args:
            state_dict: network parameters
        """
        # if fix all embedding, all no grad /  if tune, all embedding compute gradient
        if self.args.fix_embeddings:
            for p in self.network.embedding.parameters():
                p.requires_grad = False
        # all paras which need grad
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)
            
    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    # DataParallel将数据自动分割送到不同的GPU上处理，在每个模块完成工作后，DataParallel再收集整合这些结果返回。
    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)
    # Reset fixed embeddings (last part in all embeddings) to stored original value
    def reset_parameters(self):
        """Reset any partially fixed parameters to original states."""
        if self.args.tune_partial > 0:
            if self.parallel:
                embedding = self.network.module.embedding.weight.data  # all embedding
                fixed_embedding = self.network.module.fixed_embedding  # fix embedding
            else:
                embedding = self.network.embedding.weight.data
                fixed_embedding = self.network.fixed_embedding

            # Embeddings to fix are the last indices
            offset = embedding.size(0) - fixed_embedding.size(0)  # tuned partial embedding unchange (most common ques word)
            if offset >= 0:
                embedding[offset:] = fixed_embedding              # make rare word embedding fixed

    # pop fix_embedding (if in state_dict)
    def save(self, filename):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        state_dict = copy.copy(network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')  # not save  'fix embedding'
        params = {
            'state_dict': state_dict,          # model's state_dict, exclude persistent buffers. 
            'word_dict': self.word_dict,       # current word_dict class
            'feature_dict': self.feature_dict, # feature_name 2 idx ,feature_dict['in_question']=0, ['in_question_uncased']=1,['in_question_lemma']=2,['pos=NN']=3,['pos=IN']=4,['pos=DT']=5,['ner=ORG']=6,...
            'args': self.args,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')
                
    @staticmethod
    def load(filename, new_args=None,normalize_q=True,normalize_s=True):
        # new_args=args.   old model,new setting
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        args = saved_params['args']
#        # pretained model, new setting
#        if new_args:
#            # config  旧args里和新args里相同的属性，属性值替换为新arg的属性。返回替换后的旧arg
#            args = override_model_args(args, new_args)
        return DocReader(args, word_dict, feature_dict, state_dict, normalize_q, normalize_s)
    
    
    # add words (not in old model.word_dict) into word_dict(new index, new-embedding(randomly initialized))
    # new self.word_dict,self.network.embedding
    def expand_dictionary(self, words):
        # words: new words from train-dev samples' question+doc
        """Add words to the DocReader dictionary if they do not exist. The
        underlying embedding matrix is also expanded (with random embeddings).

        Args:
            words: iterable of tokens to add to the dictionary.
        Output:
            added: set of tokens that were added.
        """
        # words  not in old word_dict
        to_add = {self.word_dict.normalize(w) for w in words
                  if w not in self.word_dict}

        # Add words to dictionary and expand embedding
        if len(to_add) > 0:
            logger.info('Adding %d new words to dictionary...' % len(to_add))
            for w in to_add:
                # add w in tok2id/id2tok
                self.word_dict.add(w)
            self.args.vocab_size = len(self.word_dict)
            logger.info('New vocab size: %d' % len(self.word_dict))
 
            old_embedding = self.network.embedding.weight.data
            # new embedding, randomly initilized
            self.network.embedding = torch.nn.Embedding(self.args.vocab_size,
                                                        self.args.embedding_dim,
                                                        padding_idx=0)  # always pads padding_idx vector(initialized to zeros)
            new_embedding = self.network.embedding.weight.data
            # new embedding old words unchange         [0:|old_V|] ( in place change)
            #               new words initilized 
            new_embedding[:old_embedding.size(0)] = old_embedding

        # Return added words
        return to_add
    
    @staticmethod
    def decode(score_s, score_e, top_n=1, max_len=None):
        """Take argmax of constrained score_s * score_e.

        Args:
            score_s: independent start predictions
            score_e: independent end predictions
            top_n: number of top scored pairs to take
            max_len: max span length to consider
        """
        pred_s = []
        pred_e = []
        pred_score = []
        max_len = max_len or score_s.size(1)  # Td  / span's max_token len 
        # each ex
        # score_s: B,Td   each line is qus attention to each word in doc  (after softmax)
        # score_e: B,Td   each line is qus attention to each word in doc  (after softmax)
        for i in range(score_s.size(0)):
            # each ex i in batch
            # Outer product of scores to get full p_s * p_e matrix
            # score_s[i]: 1,T_d   start score of each token
            # score_e[i]: 1,T_d   end   score of each token
            # scores: output matrix [T_d,T_d]
            #           pos1   pos2  pos3 ... pos_Td
            #  pos1      0.1    0.2   0.3
            #  pos2      0      0.3   0.1
            #  pos3      0      0     0.3
            #  ...
            #  pos_Td
            scores = torch.ger(score_s[i], score_e[i])

            # Zero out negative length and over-length span scores
            # scores.triu_(): zuo xia 0    i<=j   j-i<max_len
            scores.triu_().tril_(max_len - 1)

            # Take argmax or top n
            scores = scores.numpy()
            # flat along line
            scores_flat = scores.flatten()
            if top_n == 1:
                idx_sort = [np.argmax(scores_flat)]   # each ex predict top_n=1    idx_sort: highest score[i,j] pos in flat score 
            elif len(scores_flat) < top_n:            # no enough span,just sort   idx_sort[0]: highest score[i,j] pos in flat score 
                idx_sort = np.argsort(-scores_flat)                              # idx_sort[1]: second highest score[i,j] pos in flat score 
            else:                                     # have enough span
                # idx: all top n unsorted span pos in flat score 
                # np.argpartition: top_n elemnet in it's right place. larger before it, smaller after it       
                idx = np.argpartition(-scores_flat, top_n)[0:top_n]  
                # -scores_flat[idx]: top n span real -score
                # np.argsort(-scores_flat[idx]): just sort these span , return sorted index in idx
                # idx_sort: top n span pos in flat score 
                idx_sort = idx[np.argsort(-scores_flat[idx])]
            # idx_sort: top 1/n/all span pos in flat score
            # scores.shape: metrix shape : T_D,T_D
            # return flat pos  along row's correspondding (i,j) in metrix
            # s_idx: tuple:(s1,s2,s3)  top n span's start pos                si:[0,T_D)
            # e_idx: tuple:(e1,e2,e3)  top n span's end   pos                ei:[0,T_D)
            # (s1,e1) span is idx_sort[0] pos in flat scores, correspong to top 1 span in this ex
            s_idx, e_idx = np.unravel_index(idx_sort, scores.shape)
            # pred_s :B,top_n    each ex's top_n start token pos
            # pred_e :B,top_n    each ex's top_n end   token pos
            pred_s.append(s_idx)
            pred_e.append(e_idx)
            # scores_flat[idx_sort]: this ex's top n span's score
            # pred_score: B,top_n    each ex's top_n span's score
            pred_score.append(scores_flat[idx_sort])
        return pred_s, pred_e, pred_score
    
    @staticmethod
    def decode_candidates(score_s, score_e, cans, top_n=1, max_len=None):
        """Take argmax of constrained score_s * score_e. Except only consider
        spans that are in the candidates list.
        """
        # cans: each ex's all can's all token spans  [2,2]    [2,4] 
        # score_s: B,T_d
        scores = []
        ans_in_can=[]
        for ex_id in range(score_s.size(0)):
            c_dict_span=dict()                       # can_index 2 spans
            for c_id in range(len(cans[ex_id])):
                c_dict_span[c_id]=cans[ex_id][c_id] 
                # print(cans[ex_id][c_id] ), c_id 's all span in tokens ([2,2]    [5,5])
            c_score=dict()                          # can 2 score
            for c_id,c_spans in c_dict_span.items():
                c_score[c_id]=0
                for s, e in c_spans:  # 2,2
                    score=score_s[ex_id][s] * score_e[ex_id][e]
                    c_score[c_id]+=score
            a,score=sorted(c_score.items(), key=lambda x:x[1],reverse=True)[0]    
            ans_in_can.append(a)
            scores.append(score)
        return ans_in_can,scores
    
    def set_normalize(self,normalize_q,normalize_s):
        self.normalize_q=normalize_q
        self.normalize_s=normalize_s
        self.network.normalize=normalize_s