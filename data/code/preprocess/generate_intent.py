#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/16 12:28
# @Author  : JJkinging
# @File    : generate_intent.py
import json

with open('../small_sample/new_B/train_final_clear_del.json', 'r', encoding='utf-8') as fp:
    raw_data = json.load(fp)

intent_write = open('../small_sample/new_B/train_intent_label.txt', 'w+', encoding='utf-8')
for filename, single_data in raw_data.items():
    intent = single_data['intent']
    intent_write.write(intent+'\n')

intent_write.close()
