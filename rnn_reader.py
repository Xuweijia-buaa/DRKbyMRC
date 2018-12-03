#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Implementation of the RNN based DrQA reader."""

import torch
import torch.nn as nn
from old_layers import SeqAttnMatch,StackedBRNN,LinearSeqAttn,BilinearSeqAttn,uniform_weights,weighted_avg


# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------


class RnnDocReader(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args, normalize=True):
        super(RnnDocReader, self).__init__()
        # Store config
        self.args = args

        # Word embeddings (+1 for padding),args from model     embedding[0]=0
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)

        # Projection for attention weighted question
        if args.use_qemb:
            # B,T_D,h,elemnet i is ith doc words to q' sum attention. dim=q_dim
            self.qemb_match = SeqAttnMatch(args.embedding_dim)

        # doc each word_embbedding size: word emb + question emb + manual features
        doc_input_size = args.embedding_dim + args.num_features
        if args.use_qemb:
            # each word's attention to q
            doc_input_size += args.embedding_dim

        # RNN document encoder
        self.doc_rnn = StackedBRNN(
            input_size=doc_input_size,                  # word_embedding + word_attention + token_feature
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,                   # rnn if padding
        )

        # RNN question encoder
        self.question_rnn = StackedBRNN(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.question_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # Output sizes of rnn encoders   2h/6h
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            doc_hidden_size *= args.doc_layers
            question_hidden_size *= args.question_layers

        # Question self-atten merging
        if args.question_merge not in ['avg', 'self_attn']:
            raise NotImplementedError('merge_mode = %s' % args.merge_mode)
        if args.question_merge == 'self_attn':
            # each word embedding * W(h,1), get this pos score (mask ,softmax)
            # B,T  each line is batch[i]'s  word self-attention score 
            self.self_attn = LinearSeqAttn(question_hidden_size)

        # Bilinear attention for span start/end
        # B,T_d , each line is qus attention to each word in doc 
        self.start_attn = BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )
        # B,T_d , each line is qus attention to each word in doc         
        self.end_attn = BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )

    def forward(self, x1, x1_f, x1_mask, x2, x2_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        # Embed both document and question
        # B,T,h
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.args.dropout_emb,
                                           training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb,
                                           training=self.training)

        # Form document encoding inputs
        # B,Td,h
        drnn_input = [x1_emb]

        # Add attention-weighted question representation
        if self.args.use_qemb:
            # B,Td,h
            # line i is ith doc words to q' sum attention. dim=q_dim
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input.append(x2_weighted_emb)

        # Add manual features
        if self.args.num_features > 0:
            # B,Td,n_f
            drnn_input.append(x1_f)

        # Encode document with RNN
        #  x        :B,T,h
        #  x_mask   :B,T    0 True, 1 false
        #  output   :B,T,2h/6h
        doc_hiddens = self.doc_rnn(torch.cat(drnn_input, 2), x1_mask)

        # Encode question with RNN + merge hiddens
        # output :B,T,2h/6h
        question_hiddens = self.question_rnn(x2_emb, x2_mask)
        
        # different word have different weight in Q
        if self.args.question_merge == 'avg':
            # B,T_q  each line : average score for each word. each word in q have same score
            # 0.5  0.5  0    0     0
            # 0.25 0.25 0.25 0.25  0
            # 0.33 0.33 0.33 0     0
            q_merge_weights = uniform_weights(question_hiddens, x2_mask)
        elif self.args.question_merge == 'self_attn':
            # B,T_q  each line: self-attention score of each word       B,T_q,2h/6h    *  W(2h/6h,1)
            # 0.2  0.53  0    0     0       
            # 0.25 0.3   0.5  0.15  0
            # 0.33 0.33 0.33  0     0
            q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        # question_hiddens B,T,2h/6h, accordding to each word
        # q_merge_weights  B,T        each ques word weight
        # each line, a vector, a weighted sum for this query :  s1*x1+s2*x2+s3*x3+...+      (1,h)
        question_hidden = weighted_avg(question_hiddens, q_merge_weights)

        # Predict start and end positions
        # B,T_d , each line is qus attention to each word in doc 
        start_scores = self.start_attn(doc_hiddens, question_hidden, x1_mask)
        # B,T_d , each line is qus attention to each word in doc    (after mask, softmaxx/not)
        end_scores = self.end_attn(doc_hiddens, question_hidden, x1_mask)
        return start_scores, end_scores
