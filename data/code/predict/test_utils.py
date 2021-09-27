#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/9 15:57
# @Author  : JJkinging
# @File    : utils.py
import torch

def load_vocab(vocab_file):
    '''construct word2id'''
    vocab = {}
    index = 0
    with open(vocab_file, 'r', encoding='utf-8') as fp:
        while True:
            token = fp.readline()
            if not token:
                break
            token = token.strip()  # 删除空白符
            vocab[token] = index
            index += 1
    return vocab


def load_reverse_vocab(vocab_file):
    '''construct id2word'''
    vocab = {}
    index = 0
    with open(vocab_file, 'r', encoding='utf-8') as fp:
        while True:
            token = fp.readline()
            if not token:
                break
            token = token.strip()  # 删除空白符
            vocab[index] = token
            index += 1
    return vocab


def collate_to_max_length(batch):
    # input_ids, input_mask
    batch_size = len(batch)
    input_ids_list = []
    input_mask_list = []
    for single_data in batch:
        input_ids_list.append(single_data[0])
        input_mask_list.append(single_data[1])

    max_length = max([len(item) for item in input_ids_list])

    output = [torch.full([batch_size, max_length],
                         fill_value=0,
                         dtype=torch.long),
              torch.full([batch_size, max_length],
                         fill_value=0,
                         dtype=torch.long)
              ]

    for i in range(batch_size):
        output[0][i][0:len(input_ids_list[i])] = torch.LongTensor(input_ids_list[i])
        output[1][i][0:len(input_mask_list[i])] = torch.LongTensor(input_mask_list[i])

    return output  # (input_ids, input_mask)
