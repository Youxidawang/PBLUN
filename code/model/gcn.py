# -*- coding: utf-8 -*-
import json, os
import math

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F



class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(768, 768))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(768))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        '''
        随机初始化参数
        :return:
        '''
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, text, adj):
        text = text.to(torch.float32)
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCNModel(nn.Module):
    def __init__(self, args):
        super(GCNModel, self).__init__()
        self.args = args
        self.gc1 = GraphConvolution(768, 768)
        self.gc2 = GraphConvolution(768, 768)
        self.gc3 = GraphConvolution(768, 768)

    def position_weight(self, x, text_len, target_len, target_tags):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        target_tags = target_tags.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_num = 0
            target_num = 0
            for j in range(1, text_len[i]):
                if target_tags[i][j] == 0:
                    context_num += 1
                else:
                    target_num += 1
            if target_num == 0 or context_num == text_len[i] - 2:
                final_weight = []
                for j in range(seq_len):
                    final_weight.append(np.float64(1))
                weight[i]=final_weight
            else:
                temp_weight = []
                context_len = seq_len - 2
                for j in range(1, text_len[i]):
                    if target_tags[i][j] != 0:
                        target_weight = np.zeros(text_len[i])
                        for k in range(1, j):
                            target_weight[k] = (1 - (j - k) / context_len)
                        right_begin = j + 1
                        for k in range(right_begin, text_len[i] - 1):
                            target_weight[k] = (1 - (k - right_begin + 1) / context_len)
                        temp_weight.append(target_weight)

                final_weight = np.zeros(seq_len)
                for j in range(text_len[i]):
                    for k in range(len(temp_weight)):
                        final_weight[j] = final_weight[j] + temp_weight[k][j]
                final_weight = final_weight / len(temp_weight)
                for j in range(text_len[i]):
                    if target_tags[i][j] != 0:
                        final_weight[j] = 0
                for j in range(seq_len):
                    weight[i].append(final_weight[j])
        weight = torch.tensor(weight).unsqueeze(2).to('cuda')
        return weight*x

    def forward(self, bert_feature, bert_tokens, target_tags, sentence_adjs, mask):
        adjs = sentence_adjs
        text_len = torch.sum(bert_tokens != 0, dim=-1)
        target_len = torch.sum(target_tags != 0, dim=-1)
        x = F.relu(self.gc1(self.position_weight(bert_feature, text_len, target_len, target_tags), adjs))
        x = F.relu(self.gc2(self.position_weight(x, text_len, target_len, target_tags), adjs))
        x = x * mask.unsqueeze(2).float().expand_as(x)

        output = x
        return output