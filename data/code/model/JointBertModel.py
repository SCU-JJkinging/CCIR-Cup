#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/8 14:40
# @Author  : JJkinging
# @File    : model.py
import torch.nn as nn
from transformers import BertModel
from data.code.model.torchcrf import CRF
import torch


class JointBertModel(nn.Module):
    def __init__(self, bert_model_path, bert_hidden_size, intent_tag_size, slot_none_tag_size, slot_tag_size, device):
        super(JointBertModel, self).__init__()
        self.bert_model_path = bert_model_path
        self.bert_hidden_size = bert_hidden_size
        self.intent_tag_size = intent_tag_size
        self.slot_none_tag_size = slot_none_tag_size
        self.slot_tag_size = slot_tag_size
        self.device = device
        self.bert = BertModel.from_pretrained(self.bert_model_path)
        self.CRF = CRF(num_tags=self.slot_tag_size, batch_first=True)

        self.intent_classification = nn.Linear(self.bert_hidden_size, self.intent_tag_size)
        self.slot_none_classification = nn.Linear(self.bert_hidden_size, self.slot_none_tag_size)
        self.slot_classification = nn.Linear(self.bert_hidden_size, self.slot_tag_size)
        self.Dropout = nn.Dropout(p=0.5)

    def forward(self, input_ids, input_mask):
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        utter_encoding = self.bert(input_ids, input_mask)
        sequence_output = utter_encoding[0]
        pooled_output = utter_encoding[1]

        pooled_output = self.Dropout(pooled_output)
        sequence_output = self.Dropout(sequence_output)

        intent_logits = self.intent_classification(pooled_output)  # [batch_size, slot_tag_size]
        slot_logits = self.slot_classification(sequence_output)
        slot_none_logits = self.slot_none_classification(pooled_output)

        return intent_logits, slot_none_logits, slot_logits

    def slot_loss(self, feats, slot_ids, mask):
        ''' 做训练时用
        :param feats: the output of BiLSTM and Liner
        :param slot_ids:
        :param mask:
        :return:
        '''
        feats = feats.to(self.device)
        slot_ids = slot_ids.to(self.device)
        mask = mask.to(self.device)
        loss_value = self.CRF(emissions=feats,
                              tags=slot_ids,
                              mask=mask,
                              reduction='mean')
        return -loss_value

    def slot_predict(self, feats, mask, id2slot):
        feats = feats.to(self.device)
        mask = mask.to(self.device)
        slot2id = {value: key for key, value in id2slot.items()}
        # 做验证和测试时用
        out_path = self.CRF.decode(emissions=feats, mask=mask)
        out_path = [[id2slot[idx] for idx in one_data] for one_data in out_path]
        for out in out_path:
            for i, tag in enumerate(out):  # tag为O、B-*、I-* 等等
                if tag.startswith('I-'):  # 当前tag为I-开头
                    if i == 0:  # 0位置应该是[START]
                        out[i] = '[START]'
                    elif out[i-1] == 'O' or out[i-1] == '[START]':  # 但是前一个tag不是以B-开头的
                        out[i] = id2slot[slot2id[tag]-1]  # 将其纠正为对应的B-开头的tag

        out_path = [[slot2id[idx] for idx in one_data] for one_data in out_path]

        return out_path
