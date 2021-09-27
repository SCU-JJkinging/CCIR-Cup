#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/8 12:39
# @Author  : JJkinging
# @File    : utils.py
import json
from torch.utils.data import Dataset


class CCFDataset(Dataset):
    def __init__(self, filename, vocab, intent_dict, slot_none_dict, slot_dict, max_length=512):
        '''
        :param filename:读取数据文件名，例如：train_seq_in.txt
        :param slot_none_dict: slot_none的字典
        :param slot_dict: slot_label的字典
        :param vocab: 词表，例如：bert的vocab.txt
        :param intent_dict: intent2id的字典
        :param max_length: 单句最大长度
        '''
        self.filename = filename
        self.vocab = vocab
        self.intent_dict = intent_dict
        self.slot_none_dict = slot_none_dict
        self.slot_dict = slot_dict
        self.max_length = max_length

        self.result = []

        # 读取数据
        with open(self.filename, 'r', encoding='utf-8') as fp:
            sen_data = fp.readlines()
        sen_data = [item.strip('\n') for item in sen_data]  # 删除句子结尾的换行符('\n')

        for utterance in sen_data:
            utterance = utterance.split(' ')  # str变list
            # 最大长度检验
            if len(utterance) > self.max_length-2:
                utterance = utterance[:max_length]

            # input_ids
            utterance = ['[CLS]'] + utterance + ['[SEP]']
            input_ids = [int(self.vocab[i]) for i in utterance]

            length = len(input_ids)

            # input_mask
            input_mask = [1] * len(input_ids)

            self.result.append((input_ids, input_mask))

    def __len__(self):
        return len(self.result)

    def __getitem__(self, index):
        input_ids, input_mask = self.result[index]

        return input_ids, input_mask
