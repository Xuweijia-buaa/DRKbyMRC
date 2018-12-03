#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Definitions of model layers/NN modules"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------


class StackedBRNN(nn.Module):
    """Stacked Bi-directional RNNs.

    Differs from standard PyTorch library in that it has the option to save
    and concat the hidden states between layers. (i.e. the output hidden size
    for each sequence input is num_layers * hidden_size).
    """

    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,  # 'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN
                 concat_layers=False, padding=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding                  # bool
        self.dropout_output = dropout_output    
        self.dropout_rate = dropout_rate        # args.dropout_rnn
        self.num_layers = num_layers            # n_layers
        self.concat_layers = concat_layers      # if concat layyer
        self.rnns = nn.ModuleList()             # model list
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size,       #  (T, B, h)  add a 1-layer bi-LSTM to model each time
                                      num_layers=1,                  #  dropout=0 /  bias=True / batch_first=False
                                      bidirectional=True))
    #  x        :B,T,h
    #  x_mask   :B,T    0 True, 1 false
    #  output   :B,T,2h/6h
    def forward(self, x, x_mask):
        """Encode either padded or non-padded sequences.

        Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.

        Args:
            x: batch * len * hdim                             B,T,h
            x_mask: batch * len (1 for padding, 0 for true)   B,T
        Output:
            x_encoded: batch * len * hdim_encoded
        """
        # all same length / not care : no  padding
        # self.padding/ 
        if x_mask.data.sum() == 0:
            # add dropout in it/ transpose B,T/ last layyer output==next layer input
            # B,T,2h/6h 
            output = self._forward_unpadded(x, x_mask)
        # eval / train+ care_len
        elif self.padding or not self.training:
            # B,T,2h/6h   padding(use packpad), and padding to original len.    mask=0, real len
            output = self._forward_padded(x, x_mask)
        else:
            # We don't care. no padding 
            # add dropout in it/ transpose B,T/ last layyer output==next layer input
            # B,T,2h/6h 
            output = self._forward_unpadded(x, x_mask)
        # return 内存连续的有相同数据的tensor
        return output.contiguous()
    
    # directly use rnn,  make output as new input, concat all output/just final output
    # no mask used
    def _forward_unpadded(self, x, x_mask):
        #  x:     B,T,h
        #  x_mask:B,T    1 false
        #  output : B,T,2h/6h
        """Faster encoding that ignores any padding."""
        # x:T,B,h: Transpose batch and sequence dims
        x = x.transpose(0, 1)
        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            # last layyer's output, as next layyer's input 
            # 1 th layyer input:    x:T,B,h==outputs[0]
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
            # Forward
            # output: T,B,2h   lstm:(output, (hn, cn)) rnn,gru:(output,hn)
            rnn_output = self.rnns[i](rnn_input)[0]
            # outputs: [x,output1,output2,output3], each is T,B,2h (except x: T,B,h)
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            # all layers: each word's all layyer embedding concat
            # T,B,2h*3
            output = torch.cat(outputs[1:], 2)
        else:
            # just final layyer: T,B,2h
            output = outputs[-1]

        # B,T,2h/6h   Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        # B,T,2h/6h
        return output

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise, encoding that handles
        padding.
        """
        #  x:     B,T,h
        #  x_mask:B,T    1 false
        #  output : B,T,2h/6h
        
        # Compute sorted sequence lengths
        # lengths: (B, )  each ex's real len: # 4,6,3,2,56,1
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        #  idx_sort: after sort, (long to short ex id in original batch:  4 1 0 2 3 5
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        
        # idx_unsort: sort (element id) to original order,   2, 1, 3, 4, 0, 5
        # id 0 elemnet should be put in first place in original batch, should get 2 pos element
        # id 1 elemnet should be put in second place in original batch, should get 1 pos element
        # resort by idx_unsort, original order
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        # sorted element(len)  for this batch : 56,6,4,3,2,1 
        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x, along batch dim, accodding to sorted_id. longest is first
        x = x.index_select(0, idx_sort)

        # x:T,B,h   Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # a PackedSequence object (must longest to shortest) ,give  sorted element and len
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                # 经过dropout等函数处理时，需要传入的是x_packed.data，是一个tensor，经过处理后，要将其重新封装成PackedSequence，再传入LSTM网络
                # hold PackedSequence input that can give to rnn model
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            # packed outputs: [x,output1,output2,output3]
            # output_pack,hn=self.rnns[i](rnn_input)
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything,enumerate i start=1  output[0] remain unchange
        for i, o in enumerate(outputs[1:], 1):
            # output[i]:T,B,2h
            # output,out_seq_len=nn.utils.rnn.pad_packed_sequence(o)
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
            # all layers: each word's all layyer embedding concat
            # T,B,2h*3
            output = torch.cat(outputs[1:], 2)
        else:
            # just final layyer: T,B,2h
            output = outputs[-1]

        # B,T,2h/6h  Transpose and unsort
        output = output.transpose(0, 1)
        # into original pos
        output = output.index_select(0, idx_unsort)

        # output max len < oroginal batch max len
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            # T dimension add paddding
            output = torch.cat([output, Variable(padding)], 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

# X:doc
# Y:ques
# B,t_x,h_y: each line i: xi to Y's weighted attention sum --> new  Y's representation  (accordding to xi)  (if word xi in Y, score will be big)
class SeqAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.

    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * hdim
        """
        # x,y all linear layyer
        if self.linear:
            # x.view(-1, x.size(2)): B*T,h                  x_proj
            # x B,T,h-->B*T,h   linear:h,h      --> B*T,h-->B,T,h
#            (0 ,.,.) =            
#              1  2  3          1  2  3
#              4  5  6          4  5  6
#              7  8  9          7  8  9
#            
#            (1 ,.,.) = 
#              1  2  3
#              4  5  6
#              7  8  9
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())   # 
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute x to y's attention scores
        # x_proj:B,Tx,h      
        # y_proj:B,h,Ty
        # x:             y.h
        #  [..x1..            [y1,y2,y3,...y_Q]
        #   ..x2..
        #   ..x3..
        #   ..x_D..]
        # score: B,T_D,T_Q
        #  [..s1..           x1 to all  [y1,y2,y3,...y_Q]    x1's attention score to Y     if  x1 in Y, some value in s1 will be big
        #   ..s2..           x2 to all  [y1,y2,y3,...y_Q]    x2's attention score to Y     if  x2 in Y, some value in s2 will be big
        #   ..s3..           x3 to all  [y1,y2,y3,...y_Q]    x3's attention score to Y                      ...
        #  ..s_D..]                      ...                        ...                                     ...
        scores = x_proj.bmm(y_proj.transpose(2, 1))

        # Mask padding
        # y_mask:  (len_q=2,3,4)
        # 0  0  1  1  1
        # 0  0  0  1  1
        # 0  0  0  0  1
        # B,Tq  --> B,1,Tq
        # (0 ,.,.) = 
        #      0  0  1  1  1
        # (1 ,.,.) = 
        #      0  0  0  1  1
        # (2 ,.,.) = 
        #      0  0  0  0  1
        # expand  : B,1,Tq  --> B,Td,Tq
        #(0 ,.,.) = 
        #  0  0  1  1  1       batch[0]: D,Q   each line is batch[0]'s ques real len
        #  0  0  1  1  1
        #  0  0  1  1  1
        #  0  0  1  1  1
        #
        #(1 ,.,.) =           batch[1]:  D,Q   each line is batch[1]'s ques real len
        #  0  0  0  1  1
        #  0  0  0  1  1
        #  0  0  0  1  1
        #  0  0  0  1  1
        #
        #(2 ,.,.) =           batch[B] :  D,Q   each line is batch[B]'s ques real len
        #  0  0  0  0  1
        #  0  0  0  0  1
        #  0  0  0  0  1
        #  0  0  0  0  1        
        y_mask = y_mask.unsqueeze(1).expand(scores.size())   # ByteTensor
        # each sample 1 pos masked into -inf
        # B,T_D,T_Q
        #(0 ,.,.) = 
        #   0.3   0.2 -inf -inf -inf      s1:   d1 to q
        #   0.1   0.8 -inf -inf -inf      s2:   d2 to q
        #   0.4   0.6 -inf -inf -inf      ...
        #   0.1   0.2 -inf -inf -inf      s_D:  d_D to q
        #
        #(1 ,.,.) = 
        #   0.1   0.2    0   -inf -inf
        #   0.3   0.2   0.3  -inf -inf
        #   0.1   0.3   0.1  -inf -inf
        #   0.1   0.4   0.1  -inf -inf
        #
        #(2 ,.,.) = 
        #   0.3   0.2   0.3   0.2 -inf
        #   0.1   0.8   0.1   0.8 -inf
        #   0.1   0.3   0.1   0.8 -inf
        #   0.1   0.4   0.1   0.8 -inf
        scores.data.masked_fill_(y_mask.data, -float('inf')) # Fills elements of self tensor with value where mask is one

        # Normalize with softmax
        # scores.view(-1, y.size(1)): B*TD, Tq. softmax to q
        alpha_flat = F.softmax(scores.view(-1, y.size(1)), dim=-1)  # dim: A dimension along which softmax will be computed.
        # B,TD,Tq
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        # Take weighted average
        # each xi's attentioned Y
        # s:  B,D,q            y : B,q,h
        #  [..s1..            [y1
        #   ..s2..             y2
        #   ..s3..             y3
        #   ..s4..             y_q]
        #   ..s_D..] 
        #  matched_seq: B,D,h
        # S1                   d1 yo q's weighted sum: S1=a1y1+a2y2+...+aTq*yTq
        # S2
        # ...
        # S_D
        matched_seq = alpha.bmm(y)
        # B,T_D,hy
        # each line i:  doc words i to q' sum attention. attention summed ques token represetation
        return matched_seq


# B,T_d , each line is qus attention to each word in doc 
class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:

    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """
    # x_size: doc_h 2h/6h
    # y_size: qus_h 2h
    def __init__(self, x_size, y_size, identity=False, normalize=True):
        super(BilinearSeqAttn, self).__init__()
        self.normalize = normalize

        # If identity is true, we just use a dot product without transformation.
        # x:doc y:ques
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        Args:
            x: batch * len * hdim1
            y: batch * hdim2
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha = batch * len
        """
        # x: B,T_d,2h/6h
        # y: B,2h           (ques vector, linear)   
        Wy = self.linear(y) if self.linear is not None else y
        # x: B,T_d,2h/6h
        # Wy:B,2h,1
        # B,T_d,1  --> B,T_d , each line is qus attention to each word in doc 
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        # B,T_d    mask by doc mask
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.normalize:
            if self.training:
                # In training we output log-softmax for NLL
                # output log-softmax
                alpha = F.log_softmax(xWy, dim=-1)
            else:
                # ...Otherwise 0-1 probabilities , test no log, just prob
                alpha = F.softmax(xWy, dim=-1)
        else:
            # exp(score), no normalize
            alpha = xWy.exp()
        # B,T_d
        return alpha

# each word embedding * W(h,1), get this pos score (mask ,softmax)
class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:

    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)
    # x     : B,T,2h/6h  qus encoder output
    # x_mask:B,T
    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        # B*T_q, 2h/6h
        x_flat = x.view(-1, x.size(-1))
        # linear: (B, T_q, 2h/6h)  -->   (B, T_q, 1) --> (B, T_q)
        # each word embedding * W(2h*1), get this word's self attention score
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        # all B,T   mask real len .  1 become -inf, 0 keep
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        # softmax
        alpha = F.softmax(scores, dim=-1)
        # B,T  each line is this ex's  word self-attention score 
        return alpha


# ------------------------------------------------------------------------------
# Functional
# ------------------------------------------------------------------------------


def uniform_weights(x, x_mask):
    """Return uniform weights over non-masked x (a sequence of vectors).

    Args:
        x: batch * len * hdim
        x_mask: batch * len (1 for padding, 0 for true)
    Output:
        x_avg: batch * hdim
    """
    # B,T_q
    alpha = Variable(torch.ones(x.size(0), x.size(1)))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    alpha = alpha / alpha.sum(1).expand(alpha.size())
    # B,T  average score for each word
    return alpha


def weighted_avg(x, weights):
    """Return a weighted average of x (a sequence of vectors).

    Args:
        x: batch * len * hdim
        weights: batch * len, sum(dim = 1) = 1
    Output:
        x_avg: batch * hdim
    """
    # x        B,T,2h/6h, accordding to each word
    # weights  B,T        each ques word weight  --> B,1,T
    # weight:  
    # B,1,T:
    # s1,s2,...sT
    # x :B,T,h
    # [x1
    #  x2
    #  x3
    #  ...
    #  xT]      
    # return: B,1,h  --> B,h
    # each line, a vector, a weighted sum for this query :  s1*x1+s2*x2+s3*x3+...+      (1,h)
    return weights.unsqueeze(1).bmm(x).squeeze(1)
