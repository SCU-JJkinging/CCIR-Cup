#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/8 12:39
# @Author  : JJkinging
# @File    : utils.py
from torch.utils.data import Dataset, DataLoader
from data.code.predict.test_utils import load_vocab, collate_to_max_length


class CCFDataset(Dataset):
    def __init__(self, filename, intent_filename, slot_filename, slot_none_filename, vocab, intent_dict,
                 slot_none_dict, slot_dict, max_length=512):
        '''
        :param filename:读取数据文件名，例如：train_seq_in.txt
        :param intent_filename: train_intent_label.txt or dev_intent_label.txt
        :param slot_filename: train_seq_out.txt
        :param slot_none_filename: train_slot_none.txt or dev_slot_none.txt
        :param slot_none_dict: slot_none的字典
        :param slot_dict: slot_label的字典
        :param vocab: 词表，例如：bert的vocab.txt
        :param intent_dict: intent2id的字典
        :param max_length: 单句最大长度
        '''
        self.filename = filename
        self.intent_filename = intent_filename
        self.slot_filename = slot_filename
        self.slot_none_filename = slot_none_filename
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

        # 读取intent
        with open(self.intent_filename, 'r', encoding='utf-8') as fp:
            intent_data = fp.readlines()
        intent_data = [item.strip('\n') for item in intent_data]  # 删除结尾的换行符('\n')
        intent_ids = [intent_dict[item] for item in intent_data]

        # 读取slot_none
        with open(self.slot_none_filename, 'r', encoding='utf-8') as fp:
            slot_none_data = fp.readlines()
        # 删除结尾的空格和换行符('\n')
        slot_none_data = [item.strip('\n').strip(' ').split(' ') for item in slot_none_data]
        # 下面列表表达式把slot_none转为id
        slot_none_ids = [[self.slot_none_dict[ite] for ite in item] for item in slot_none_data]

        # 读取slot
        with open(self.slot_filename, 'r', encoding='utf-8') as fp:
            slot_data = fp.readlines()
        slot_data = [item.strip('\n') for item in slot_data]  # 删除句子结尾的换行符('\n')
        # slot_ids = [self.slot_dict[item] for item in slot_data]

        idx = 0
        for utterance in sen_data:
            utterance = utterance.split(' ')  # str变list
            slot_utterence = slot_data[idx].split(' ')
            # 最大长度检验
            if len(utterance) > self.max_length-2:
                utterance = utterance[:max_length]
                slot_utterence = slot_utterence[:max_length]

            # input_ids
            utterance = ['[CLS]'] + utterance + ['[SEP]']
            input_ids = [int(self.vocab[i]) for i in utterance]


            length = len(input_ids)

            # slot_ids
            slot_utterence = ['[START]'] + slot_utterence + ['[EOS]']
            slot_ids = [int(self.slot_dict[i]) for i in slot_utterence]

            # input_mask
            input_mask = [1] * len(input_ids)

            # intent_ids
            intent_id = intent_ids[idx]

            # slot_none_ids
            slot_none_id = slot_none_ids[idx]  # slot_none_id 为 int or list

            idx += 1

            self.result.append((input_ids, slot_ids, input_mask, intent_id, slot_none_id))

    def __len__(self):
        return len(self.result)

    def __getitem__(self, index):
        input_ids, slot_ids, input_mask, intent_id, slot_none_id = self.result[index]

        return input_ids, slot_ids, input_mask, intent_id, slot_none_id


if __name__ == "__main__":
    filename = '../dataset/final_data/train_seq_in.txt'
    vocab_file = '../dataset/pretrained_model/erine/vocab.txt'
    intent_filename = '../dataset/final_data/train_intent_label.txt'
    slot_filename = '../dataset/final_data/train_seq_out.txt'
    slot_none_filename = '../dataset/final_data/train_slot_none.txt'
    intent_label = '../dataset/final_data/intent_label.txt'
    slot_label = '../dataset/final_data/slot_label.txt'
    slot_none_vocab = '../dataset/final_data/slot_none_vocab.txt'
    intent_dict = load_vocab(intent_label)
    slot_dict = load_vocab(slot_label)
    slot_none_dict = load_vocab(slot_none_vocab)
    vocab = load_vocab(vocab_file)

    dataset = CCFDataset(filename, intent_filename, slot_filename, slot_none_filename, vocab, intent_dict,
                         slot_none_dict, slot_dict)

    dataloader = DataLoader(dataset, shuffle=False, batch_size=8, collate_fn=collate_to_max_length)

    for batch in dataloader:
        print(batch)
        break
