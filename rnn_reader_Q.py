#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 20:20:03 2018

@author: xuweijia
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import torch.nn.utils.rnn.pack_padded_sequence as pack
#import torch.nn.utils.rnn.PackedSequence as repack
#import torch.nn.utils.rnn.pad_packed_sequence  as unpack
from torch.autograd import Variable
from utils import to_var,isnan
class RnnDocReader_Q(nn.Module):
    def __init__(self, args, normalize=True):
        super(RnnDocReader_Q, self).__init__()
        RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
        self.rnn_type=args.rnn_type
        rnn=RNN_TYPES[self.rnn_type]
        self.self_linear=True
        self.concat=args.concat_layers
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size,args.embedding_dim,padding_idx=0)  # embedding[0]=0
        self.doc_input_size = args.embedding_dim                                         # each token: h + n_features value
        self.h=args.hidden_size
        self.dropout_rnn=args.dropout_rnn                 # dropout between  layyers
        self.final_dropout=args.dropout_rnn_output    # dropout of final rnn output (last layyer)
        self.h_output=0.4
        self.embed_size=args.embedding_dim
        if self.args.doc_use_qemb:                    # doc to q's attention, as doc token's feature
            self.doc_input_size+=args.embedding_dim
            if self.self_linear:
                self.Linear_self=nn.Linear(args.embedding_dim,self.embed_size)
        if self.args.num_features>0:
            self.doc_input_size+=self.args.num_features
        # doc encoder : multi layyer/ 1 layyer
        self.doc_encoder=nn.ModuleList()
        for i in (range(args.doc_layers)):
            input_size=self.doc_input_size if i==0 else 2*self.h
            doc_rnn=rnn(input_size,self.h,batch_first=True,num_layers=1,bidirectional=True) # one layyer no need dropout (just between layyers and except final)
            self.doc_encoder.append(doc_rnn)
        # ques encoder : multi layyer/ 1 layyer
        self.ques_encoder=nn.ModuleList()
        for i in (range(args.ques_layers)):
            input_size=args.embedding_dim if i==0 else 2*self.h
            ques_rnn=rnn(input_size,self.h,batch_first=True,num_layers=1,bidirectional=True) # one layyer no need dropout (just between layyers and except final)
            self.ques_encoder.append(ques_rnn)
        self.ques_output_size=2 *self.h * args.ques_layers if self.concat else 2 *self.h
        if args.q_self_weight:                   # get B,|Q|, each word in q get it's weight
            self.q_self_Linear=nn.Linear(self.ques_output_size,1)
        # Q encoder : multi layyer/ 1 layyer
        self.Q_encoder=nn.ModuleList()
        for i in (range(args.Q_layers)):
            input_size=self.args.embedding_dim if i==0 else 2*self.h
            Q_rnn=rnn(input_size,self.h,batch_first=True,num_layers=1,bidirectional=True) # one layyer no need dropout (just between layyers and except final)
            self.Q_encoder.append(Q_rnn)

        self.doc_output_size=2 *self.h * args.doc_layers if self.concat else 2 *self.h             # n_lay * n_direction *h 
        self.Q_output_size=2 *self.h * args.Q_layers if self.concat else 2 *self.h                 # n_lay * n_direction *h
        
        if args.train_mode=='string_match_base_dis' or args.train_mode=='span':
            self.q2doc_s=nn.Linear(self.ques_output_size,self.doc_output_size)
            self.q2doc_e=nn.Linear(self.ques_output_size,self.doc_output_size)
        elif args.train_mode=='contain' or args.train_mode=='NER':
            self.q2doc=nn.Linear(self.ques_output_size,self.doc_output_size)
            
        self.Q2doc=nn.Linear(self.Q_output_size,self.doc_output_size)
        self.normalize=normalize
            
    def forward(self, dw, f, dw_mask, qw, qw_mask, Qw,Qw_mask,Q_mask,ex2Q):
        # embeddings
        dw_emb = self.embedding(dw) # B,|D|,h
        qw_emb = self.embedding(qw) # B,|Q|,h
        Qw_emb = self.embedding(Qw) # Q_max,|Q_tokens|,h
        B=len(dw_emb)
        # Q=len(Qw_emb)
        # dropout on embeddings
        if self.args.dropout_emb > 0:
            dw_emb = F.dropout(dw_emb, p=self.args.dropout_emb,training=self.training)
            qw_emb = F.dropout(qw_emb, p=self.args.dropout_emb,training=self.training)
            Qw_emb = F.dropout(Qw_emb, p=self.args.dropout_emb,training=self.training)
        # each doc token's att sum vector for query, as this token's soft feature vector (compare with q_in_token)
        doc_input=[dw_emb]
        if self.args.doc_use_qemb:
            if self.self_linear:
                dw_project=self.Linear_self(dw_emb.view(-1,self.embed_size)).view(B,-1,self.embed_size)   # B,|D|,h
                dw_project=F.relu(dw_project)
                qw_project=self.Linear_self(qw_emb.view(-1,self.embed_size)).view(B,-1,self.embed_size)   # B,|Q|,h
                qw_project=F.relu(qw_project)
            else:
                dw_project=dw_emb
                qw_project=qw_emb
            b2q_att=torch.bmm(dw_project,qw_project.transpose(2, 1))                               # B,|D|,|Q|, each d to all q's attention score
            b2q_att=b2q_att.clone()
            b2q_att[qw_mask.unsqueeze(1).expand_as(b2q_att).data]=-float('inf')
            # b2q_att.data.masked_fill_(qw_mask.unsqueeze(1).expand_as(b2q_att).data,-float('inf'))  # masked with q's real len
            b2q_att=F.softmax(b2q_att.view(-1,qw_emb.size(1))).view(B,dw_project.size(1),qw_project.size(1)) # and softmax  B,|D|,|Q|      0.1,0.3,0.6,0
            b2q_each_vec=torch.bmm(b2q_att,qw_project)                                             # B,|D|,h_q   each d's summed attention to Q: 1,h
            doc_input.append(b2q_each_vec)
        if self.args.num_features>0:
            doc_input.append(f)
         
    # doc encoder
        #  B,|D|,h,  (B,|D|,h),  B,|D|,n_f 
        doc_input=torch.cat(doc_input,2)
        # no padding
        if (self.training and not self.args.rnn_padding) or dw_mask.data.sum() == 0:
            outputs=[doc_input]
            hns=[]
            for i in range(len(self.doc_encoder)):
                inputs=outputs[-1]
                # dropout on this layyer
                inputs=F.dropout(inputs,training=self.training,p=self.dropout_rnn)
                output,h_n=self.doc_encoder[i](inputs) # output: B,T,n_direction*n_h_dden  # h_n:n_direction,B,n_h_dden    # lstm hn:(h_n, c_n)
                outputs.append(output)
                h_n=torch.cat(h_n,-1) if self.rnn_type!='lstm' else torch.cat(h_n[0],-1)
                hns.append(h_n)
            if self.concat:
                doc_output=torch.cat(outputs[1:],-1)   # B,D, n_lay*n_direct*h   each token t :  h1,t->,h1,t <-, h2,t->,h2,t <-,  h3,t->,h3,t <-,
            else:
                doc_output=outputs[-1]                 # B,D, n_direct*h         each token t:   h3,t->,h3,t <-,
       # padding
        elif self.args.rnn_padding or not self.training:   
            l=torch.sum(dw_mask.eq(0).long(),1).squeeze(-1)       # B, real len
            sort_len,sort_idx=torch.sort(l,dim=0,descending=True) # B,
            _,resort=torch.sort(sort_idx,dim=0)                   # resort B's ex to original
            outputs=[doc_input[sort_idx.data]]
            hns=[]                   
            for i in range(len(self.doc_encoder)): 
                inputs=outputs[-1]
                pack_inputs=torch.nn.utils.rnn.pack_padded_sequence(inputs,sort_len.data.cpu().numpy(), batch_first=True)  # pack input .  len: numpy/list
                inputs=F.dropout(pack_inputs.data,training=self.training,p=self.dropout_rnn) # dropout
                inputs=torch.nn.utils.rnn.PackedSequence(inputs,pack_inputs.batch_sizes)                           # repack
                output,h_n=self.doc_encoder[i](inputs) # output: B,T,n_direction*n_h_dden   # h_n:    n_direction,B,n_h_dden
                output,_=torch.nn.utils.rnn.pad_packed_sequence(output,batch_first=True)                   # real_output, output_len
                outputs.append(output)
                h_n=torch.cat(h_n,-1) if self.rnn_type!='lstm' else torch.cat(h_n[0],-1)
                hns.append(h_n)
            if self.concat:
                doc_output=torch.cat(outputs[1:],-1)   # B,D, n_lay*n_direct*h   each token t :  h1,t->,h1,t <-, h2,t->,h2,t <-,  h3,t->,h3,t <-,
            else:
                doc_output=outputs[-1]                 # B,D, n_direct*h         each token t:   h3,t->,h3,t <-,    
            doc_output=doc_output[resort.data]
            # after padding, doc len may shorter,# padding on some dimension in t
            if doc_output.size(1) != dw_mask.size(1):
                padding = torch.zeros(doc_output.size(0),dw_mask.size(1) -doc_output.size(1),doc_output.size(2)).type(doc_output.data.type())
                doc_output = torch.cat([doc_output, Variable(padding)], 1)
        if self.concat:
            doc_h=torch.cat(hns,-1)   # B,n_direc*n_layyer*h
        else:
            doc_h=hns[-1]             # B,n_direc*h
            
        doc_output=F.dropout(doc_output,training=self.training,p=self.final_dropout)   # B,|D|,n_direc*n_layyer*h /  B,|D|,n_direc*h            
        doc_h=F.dropout(doc_h,training=self.training,p=self.h_output)                  # B,n_direc*n_layyer*h / B,n_direc*h
        
    # question encoder
        # no padding
        if (self.training and not self.args.rnn_padding) or qw_mask.data.sum() == 0:
            outputs=[qw_emb]  # B,|Q|,h
            hns=[]
            for i in range(len(self.ques_encoder)):
                inputs=outputs[-1]
                # dropout on this layyer
                inputs=F.dropout(inputs,training=self.training,p=self.dropout_rnn)
                output,h_n=self.ques_encoder[i](inputs) # output: B,T,n_direction*n_h_dden  # h_n:n_direction,B,n_h_dden    # lstm hn:(h_n, c_n)
                #print(output.size())
                outputs.append(output)
                h_n=torch.cat(h_n,-1) if self.rnn_type!='lstm' else torch.cat(h_n[0],-1)
                hns.append(h_n)
            if self.concat:
                ques_output=torch.cat(outputs[1:],-1)   # B,Q, n_lay*n_direct*h   each token t :  h1,t->,h1,t <-, h2,t->,h2,t <-,  h3,t->,h3,t <-,
            else:
                ques_output=outputs[-1]                 # B,Q, n_direct*h         each token t:   h3,t->,h3,t <-,
       # padding
        elif self.args.rnn_padding or not self.training:   
            l=torch.sum(qw_mask.eq(0).long(),1).squeeze(-1)       # B, real len
            sort_len,sort_idx=torch.sort(l,dim=0,descending=True) # B,
            _,resort=torch.sort(sort_idx,dim=0)                   # resort B's ex to original
            outputs=[qw_emb[sort_idx.data]]
            hns=[]      
            for i in range(len(self.ques_encoder)): 
                inputs=outputs[-1]
                pack_inputs=torch.nn.utils.rnn.pack_padded_sequence(inputs,sort_len.data.cpu().numpy(), batch_first=True)   # pack input .  len: numpy/list
                inputs=F.dropout(pack_inputs.data,training=self.training,p=self.dropout_rnn) # dropout
                inputs=torch.nn.utils.rnn.PackedSequence(inputs,pack_inputs.batch_sizes)                           # repack
                output,h_n=self.ques_encoder[i](inputs) # output: B,T,n_direction*n_h_dden   # h_n:    n_direction,B,n_h_dden
                output,_=torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)            # real_output, output_len
                #print(output.size())
                outputs.append(output)
                h_n=torch.cat(h_n,-1) if self.rnn_type!='lstm' else torch.cat(h_n[0],-1)
                hns.append(h_n)
            if self.concat:
                ques_output=torch.cat(outputs[1:],-1)   # B,T, n_lay*n_direct*h   each token t :  h1,t->,h1,t <-, h2,t->,h2,t <-,  h3,t->,h3,t <-,
            else:
                ques_output=outputs[-1]                 # B,T, n_direct*h         each token t:   h3,t->,h3,t <-,    
            ques_output=ques_output[resort.data]
            # after padding, doc len may shorter,# padding on some dimension in t
            if ques_output.size(1) != qw_mask.size(1):
                padding = torch.zeros(ques_output.size(0),qw_mask.size(1) -ques_output.size(1),ques_output.size(2)).type(ques_output.data.type())
                ques_output = torch.cat([ques_output, Variable(padding)], 1)
        if self.concat:
            ques_h=torch.cat(hns,-1)   # B,n_direc*n_layyer*h
        else:
            ques_h=hns[-1]             # B,n_direc*h
            
        ques_output=F.dropout(ques_output,training=self.training,p=self.final_dropout)   # B,|Q|,n_direc*n_layyer*h /  B,|Q|,n_direc*h            
        ques_h=F.dropout(ques_h,training=self.training,p=self.h_output)                  # B,q_h:   B,n_direc*n_layyer*h/  B,n_direc*h
        # give different q_token different weight
        if self.args.q_self_weight: 
            #print(self.h)
            #print(self.ques_output_size)
            #print(ques_output.size())  # B,T, n_lay*n_direct*h 
            self_score=self.q_self_Linear(ques_output.view(-1,self.ques_output_size)).squeeze(-1).view(B,-1)       # B*|Q|,h    h,1 -->  B,Q   each q token's self score
            self_score=self_score.clone()
            self_score[qw_mask.data]=-float('inf')
            #self_score.data.masked_fill_(qw_mask.data,-float('inf'))
            self_score=F.softmax(self_score)    # B,|Q|
            ques_final=torch.bmm(self_score.unsqueeze(1),ques_output).squeeze(1)         # B,1,|Q|  * B,|Q|,q_h  --> B,q_h  can use ques_final/ ques_h

    # Q encoder              
        # no padding
        if (self.training and not self.args.rnn_padding) or Qw_mask.data.sum() == 0:
            outputs=[Qw_emb]  # n_Q,|Q_tokens|,h
            hns=[]
            for i in range(len(self.Q_encoder)):
                inputs=outputs[-1]
                # dropout on this layyer
                inputs=F.dropout(inputs,training=self.training,p=self.dropout_rnn)
                output,h_n=self.Q_encoder[i](inputs) # output: B,T,n_direction*n_h_dden  # h_n:n_direction,B,n_h_dden    # lstm hn:(h_n, c_n)
                outputs.append(output)
                h_n=torch.cat(h_n,-1) if self.rnn_type!='lstm' else torch.cat(h_n[0],-1)
                hns.append(h_n)
            if self.concat:
                Q_output=torch.cat(outputs[1:],-1)   # B,Q, n_lay*n_direct*h   each token t :  h1,t->,h1,t <-, h2,t->,h2,t <-,  h3,t->,h3,t <-,
            else:
                Q_output=outputs[-1]                 # B,Q, n_direct*h         each token t:   h3,t->,h3,t <-,
       # padding
        elif self.args.rnn_padding or not self.training:  
            l=torch.sum(Qw_mask.eq(0).long(),1).squeeze(-1)       # B, real len
            sort_len,sort_idx=torch.sort(l,dim=0,descending=True) # B,
            _,resort=torch.sort(sort_idx,dim=0)                   # resort B's ex to original
            outputs=[Qw_emb[sort_idx.data]]
            hns=[]
            for i in range(len(self.Q_encoder)): 
                inputs=outputs[-1]
                pack_inputs=torch.nn.utils.rnn.pack_padded_sequence(inputs,sort_len.data.cpu().numpy(), batch_first=True)   # pack input .  len: numpy/list
                inputs=F.dropout(pack_inputs.data,training=self.training,p=self.dropout_rnn) # dropout
                inputs=torch.nn.utils.rnn.PackedSequence(inputs,pack_inputs.batch_sizes)                           # repack
                output,h_n=self.Q_encoder[i](inputs) # output: B,T,n_direction*n_h_dden   # h_n:    n_direction,B,n_h_dden
                output,_=torch.nn.utils.rnn.pad_packed_sequence(output,batch_first=True)            # real_output, output_len
                outputs.append(output)
                h_n=torch.cat(h_n,-1) if self.rnn_type!='lstm' else torch.cat(h_n[0],-1)
                hns.append(h_n)
            if self.concat:
                Q_output=torch.cat(outputs[1:],-1)   # B,T, n_lay*n_direct*h   each token t :  h1,t->,h1,t <-, h2,t->,h2,t <-,  h3,t->,h3,t <-,
            else:
                Q_output=outputs[-1]                 # B,T, n_direct*h         each token t:   h3,t->,h3,t <-,    
            Q_output=Q_output[resort.data]
            # after padding, doc len may shorter,# padding on some dimension in t
            if Q_output.size(1) != Qw_mask.size(1):
                padding = torch.zeros(Q_output.size(0),Qw_mask.size(1)-Q_output.size(1),Q_output.size(2)).type(Q_output.data.type())
                Q_output = torch.cat([Q_output, Variable(padding)], 1)
        if self.concat:
            Q_h=torch.cat(hns,-1)   # |n_Q|,n_direc*n_layyer*h
        else:
            Q_h=hns[-1]             # |n_Q|,n_direc*h
            
        Q_output=F.dropout(Q_output,training=self.training,p=self.final_dropout)   # n_Q,|Q_tokens|,n_direc*n_layyer*h /  n_Q,|Q_tokens|,n_direc*h            
        Q_h=F.dropout(Q_h,training=self.training,p=self.h_output)                  # n_Q, n_direc*n_layyer*h/  n_Q ,n_direc*h        

    # Q2d    
        wQ=self.Q2doc(Q_h)                                       # n_Q,h_Q *    h_Q,h_d --> n_Q, h_d
        #print(type(Q_mask))
        trans_Q=np.zeros([B,Q_mask.size(1),self.doc_output_size],dtype='float32')                                    
        trans_Q=to_var(trans_Q,self.args.cuda)           # B,max_Q,h_d     *   doc B,h,1  -->   B,max_Q, with mask  
        for ex_id in range(B):
            start,end=ex2Q[ex_id]          # Q's pos range in all Q
            trans_Q[ex_id,:end-start,:]=wQ[start:end,:].clone()
        pure_Q=torch.bmm(trans_Q,doc_h.unsqueeze(2)).squeeze(2)  # B,max_Q   Q_mask,same size      Q*W*D
    # q 2 each d 
        ques_final=ques_h  # ques_h /  ques_final    B,h_q --> B,1,h_d
        if self.args.train_mode=='string_match_base_dis':
            score_s=torch.bmm(self.q2doc_s(ques_final).unsqueeze(1),doc_output.transpose(1,2)).squeeze(1)  #  B,1,h_d * B,h_d,|D| --> B,1,|D|-->  B,|D|
            score_e=torch.bmm(self.q2doc_e(ques_final).unsqueeze(1),doc_output.transpose(1,2)).squeeze(1)  #  B,1,h_d * B,h_d,|D| --> B,1,|D|-->  B,|D|
            
            score_s=score_s.clone()
            score_s[dw_mask.data]=-float('inf')
            score_e=score_e.clone()
            score_e[dw_mask.data]=-float('inf')

            # score_s.data.masked_fill_(dw_mask.data,-float('inf'))
            # score_e.data.masked_fill_(dw_mask.data,-float('inf'))
            if self.training:
                score_s=F.log_softmax(score_s)  # B,|D|, to compute B,max_C
                score_e=F.log_softmax(score_e)  # B,|D|, to compute B,max_C
            else:
                score_s=torch.exp(score_s)             # B,|D| 
                score_e=torch.exp(score_e)             # B,|D|
            #print(score_e)
            #print(score_s)
            assert torch.sum(isnan(score_s.data.cpu()))==0
            assert torch.sum(isnan(score_e.data.cpu()))==0
            return score_s,score_e,pure_Q              # pure_Q   B,max_Q, before mask 
              
        if self.args.train_mode=='span':
            score_s=torch.bmm(self.q2doc_s(ques_final).unsqueeze(1),doc_output.transpose(1,2)).squeeze(1)  #  B,1,h_d * B,h_d,|D| --> B,1,|D|-->  B,|D|
            score_e=torch.bmm(self.q2doc_e(ques_final).unsqueeze(1),doc_output.transpose(1,2)).squeeze(1)  #  B,1,h_d * B,h_d,|D| --> B,1,|D|-->  B,|D|
            #print(ques_final)  # hd:768    D:   B:64
            #print(doc_output)  # ~
            #print(score_s)
            #print(score_e)
            # print({'score_e_before_mask':score_e})
            # print({'dw_mask':dw_mask})
            score_s=score_s.clone()
            score_s[dw_mask.data]=-float('inf')
            score_e=score_e.clone()
            score_e[dw_mask.data]=-float('inf')
            #print(score_e)
            #print(score_s)
            # print({'score_e_before_softmax':score_e})
            #score_s.data.masked_fill_(dw_mask.data,-float('inf'))
            #score_e.data.masked_fill_(dw_mask.data,-float('inf'))  
            if self.training or self.normalize:
                score_s=F.softmax(score_s)
                score_e=F.softmax(score_e)
            else:                   
                score_s=torch.exp(score_s)             # B,|D| 
                score_e=torch.exp(score_e)             # B,|D| 
            # print({'score_e_after_softmax':score_e})
            #print(score_e)
            #print(score_s)
            # assert torch.sum(isnan(score_e.data.cpu()))==0
            # assert torch.sum(isnan(score_s.data.cpu()))==0
            return score_s,score_e,pure_Q #,ques_final,doc_output
            
        if self.args.train_mode=='contain' or self.args.train_mode=='NER':
            score=torch.bmm(self.q2doc(ques_final).unsqueeze(1),doc_output.transpose(1,2)).squeeze(1)       #  B,1,h_d * B,h_d,|D| --> B,1,|D|-->  B,|D|
            # score.data.masked_fill_(dw_mask.data,-float('inf'))
            score=score.clone()
            score[dw_mask.data]=-float('inf')
            if self.training or self.normalize:
                score=F.softmax(score)
            else:
                score=torch.exp(score)
            assert torch.sum(isnan(score.data.cpu()))==0
            return score,pure_Q
                
        
 
            
        


        
                    
            
            
            
                
