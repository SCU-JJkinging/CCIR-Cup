#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/12 18:52
# @Author  : JJkinging
# @File    : slot_sorted.py

'''对train.json的数据按slot的value的长度从大到小排序'''
import json

with open('../small_sample/new_B/train_final_clear.json', 'r', encoding='utf-8') as fp:
    raw_data = json.load(fp)
res = {}
for filename, single_data in raw_data.items():
    tmp_dict = {}
    tmp_dict['text'] = single_data['text']
    tmp_dict['intent'] = single_data['intent']
    slots = single_data['slots']
    dic = {}
    tmp_tuple = sorted(slots.items(), key=lambda x: len(x[1]), reverse=True)
    for tuple_data in tmp_tuple:
        dic[tuple_data[0]] = tuple_data[1]
    tmp_dict['slots'] = dic
    res[filename] = tmp_dict

with open('../small_sample/new_B/train_final_clear_sorted.json', 'w', encoding='utf-8') as fp:
    json.dump(res, fp, ensure_ascii=False)
