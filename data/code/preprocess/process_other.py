#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/10 22:22
# @Author  : JJkinging
# @File    : process_other.py
import json

with open('../../dataset/final_data/other_2.txt', 'r', encoding='utf-8') as fp:
    data = fp.readlines()
    data = [item.split(' ')[0].strip('\n') for item in data]


res = {}
for i, sen in enumerate(data):
    tem_dict = {}
    tem_dict['text'] = sen
    tem_dict['intent'] = 'Other'
    tem_dict['slots'] = {}

    lens = len(str(i))
    o_nums = 5 - lens
    filename = 'NLU' + '0' * o_nums + str(i)
    res[filename] = tem_dict


with open('../../dataset/small_sample/data/other_2.json', 'w', encoding='utf-8') as fp:
    json.dump(res, fp, ensure_ascii=False)
