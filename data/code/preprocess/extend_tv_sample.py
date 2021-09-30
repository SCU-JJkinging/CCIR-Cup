#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/16 11:18
# @Author  : JJkinging
# @File    : extend_tv_sample.py
'''数据增强: 扩充TVProgram-Play小样本数据集'''
import json
import random
random.seed(1000)

with open('../../small_sample/same_intent/TVProgram-Play.json', 'r', encoding='utf-8') as fp:
    ori_data = json.load(fp)

with open('../../small_sample/tv_entity_dic.json', 'r', encoding='utf-8') as fp:
    tv_dic = json.load(fp)

res = {}
idx = 0
for filename, single_data in ori_data.items():
    text = single_data['text']
    slots = single_data['slots']
    for _ in range(30):  # 每条扩充99条数据
        tmp_dic = {}
        slot_dic = {}
        length = len(str(idx))
        o_num = 5 - length
        new_filename = 'NLU' + '0'*o_num + str(idx)
        idx += 1

        new_text = text
        for prefix in tv_dic['prefix']:
            if text[:-3].find(prefix) != -1:
                ran_i = random.randint(0, len(tv_dic['prefix'])-1)
                new_text = text.replace(prefix, tv_dic['prefix'][ran_i])  # 随机找一个prefix替换
                break
            else:
                ran_i = random.randint(0, len(tv_dic['prefix'])-1)
                new_text = tv_dic['prefix'][ran_i] + text
        if 'name' in slots.keys():
            ran_i = random.randint(0, len(tv_dic['name'])-1)
            new_text = new_text.replace(slots['name'], tv_dic['name'][ran_i])
            slot_dic['name'] = tv_dic['name'][ran_i]  # 随机替换name
        if 'channel' in slots.keys():
            ran_i = random.randint(0, len(tv_dic['channel']) - 1)
            new_text = new_text.replace(slots['channel'], tv_dic['channel'][ran_i])
            slot_dic['channel'] = tv_dic['channel'][ran_i]  # 随机替换name
        if 'datetime_date' in slots.keys():
            ran_i = random.randint(0, len(tv_dic['datetime_date']) - 1)
            new_text = new_text.replace(slots['datetime_date'], tv_dic['datetime_date'][ran_i])
            slot_dic['datetime_date'] = tv_dic['datetime_date'][ran_i]  # 随机替换language
        if 'datetime_time' in slots.keys():
            ran_i = random.randint(0, len(tv_dic['datetime_time']) - 1)
            new_text = new_text.replace(slots['datetime_time'], tv_dic['datetime_time'][ran_i])
            slot_dic['datetime_time'] = tv_dic['datetime_time'][ran_i]  # 随机替换language
        tmp_dic['text'] = new_text
        tmp_dic['intent'] = "TVProgram-Play"
        tmp_dic['slots'] = slot_dic
        res[new_filename] = tmp_dic


print(res)
print(len(res))

with open('../../extend_data/data/extend_tv.json', 'w', encoding='utf-8') as fp:
    json.dump(res, fp, ensure_ascii=False)

