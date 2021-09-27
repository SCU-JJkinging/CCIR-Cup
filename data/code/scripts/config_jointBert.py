#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/8 15:13
# @Author  : JJkinging
# @File    : config.py
class Config(object):
    '''配置类'''

    def __init__(self):
        self.train_file = '../../user_data/train_data/train_seq_in.txt'
        self.dev_file = '../../user_data/dev_data/dev_seq_in.txt'
        self.test_file = '../../user_data/test_data/test_seq_in.txt'
        self.intent_label_file = '../../user_data/common_data/intent_label.txt'
        self.vocab_file = '../../user_data/pretrained_model/erine/vocab.txt'
        self.train_intent_file = '../../user_data/train_data/train_intent_label.txt'
        self.dev_intent_file = '../../user_data/dev_data/dev_intent_label.txt'
        self.max_length = 512
        self.batch_size = 16
        self.test_batch_size = 32
        self.bert_model_path = '../../user_data/pretrained_model/erine'
        self.checkpoint = None  # '../../user_data/output_model/JointBert/trained_model/model_42.pth.tar'
        self.use_gpu = True
        self.cuda = "cuda:0"
        self.attention_dropout = 0.1
        self.bert_hidden_size = 768
        self.embedding_dim = 300
        self.lr = 5e-5  # 5e-5
        self.crf_lr = 5e-2  # 5e-2
        self.weight_decay = 0.0
        self.epochs = 60
        self.max_grad_norm = 4
        self.patience = 60
        self.target_dir = '../../user_data/output_model/JointBert'
        self.slot_none_vocab = '../../user_data/common_data/slot_none_vocab.txt'
        self.slot_label = '../../user_data/common_data/slot_label.txt'
        self.train_slot_filename = '../../user_data/train_data/train_seq_out.txt'
        self.dev_slot_filename = '../../user_data/dev_data/dev_seq_out.txt'
        self.train_slot_none_filename = '../../user_data/train_data/train_slot_none.txt'
        self.dev_slot_none_filename = '../../user_data/dev_data/dev_slot_none.txt'

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


if __name__ == '__main__':
    con = Config()
    con.update(gpu=8)
    print(con.gpu)
    print(con)
