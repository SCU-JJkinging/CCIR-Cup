#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/15 21:07
# @Author  : JJkinging
# @File    : extract_small_sample.py
import json

with open('../raw_data/train_8.json', 'r', encoding='utf-8') as fp:
    raw_data = json.load(fp)

res = {}
for filename, single_data in raw_data.items():
    intent = single_data['intent']
    if intent == 'TVProgram-Play':
        res[filename] = single_data

with open('../extend_data/same_intent/ori_data/TVProgram-Play.json', 'w', encoding='utf-8') as fp:
    json.dump(res, fp, ensure_ascii=False)
