#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/20 18:35
# @Author  : JJkinging
# @File    : integration.py
import json

with open('../result/JointBert/new_B/result_bert_42_post.json', 'r', encoding='utf-8') as fp:
    second_data = json.load(fp)
with open('../result/InteractModel/1/new_B/result_interact_1_post.json', 'r', encoding='utf-8') as fp:
    third_data = json.load(fp)
with open('../result/InteractModel/3/new_B/result_interact_3_post.json', 'r', encoding='utf-8') as fp:
    best_data = json.load(fp)

idx_list = []
res = {}
for i in range(len(best_data)):
    lengths = len(str(i))
    o_nums = 5 - lengths
    idx = 'NLU' + '0'*o_nums + str(i)
    tmp = {}

    best_dic = best_data[idx]
    second_dic = second_data[idx]
    third_dic = third_data[idx]
    # 调整intent
    if best_dic['intent'] == second_dic['intent'] == third_dic['intent'] or \
            best_dic['intent'] == second_dic['intent'] or best_dic['intent'] == third_dic['intent'] or \
            (best_dic['intent'] != second_dic['intent'] and best_dic['intent'] != third_dic['intent'] and
             second_dic['intent'] != third_dic['intent']):
        intent = best_dic['intent']

    elif second_dic['intent'] == third_dic['intent']:
        intent = second_dic['intent']

    slot_dic = {}
    # 调整slot
    best_slots = best_dic['slots']
    second_slots = second_dic['slots']
    third_slots = third_dic['slots']

    total_slot_keys = set()
    total_slot_keys.update(list(best_slots.keys()))
    total_slot_keys.update(list(second_slots.keys()))
    total_slot_keys.update(list(third_slots.keys()))

    second_key = []
    third_key = []
    for key in total_slot_keys:
        if key in best_slots.keys() and key in second_slots.keys() and key in third_slots.keys():  # 若三者都有
            if isinstance(best_slots[key], list) and isinstance(second_slots[key], list) and \
                    isinstance(third_slots[key], list):
                best_slots[key] = set(best_slots[key])
                second_slots[key] = set(second_slots[key])
                third_slots[key] = set(third_slots[key])

            if best_slots[key] == second_slots[key] == third_slots[key] or \
                    best_slots[key] == second_slots[key] or best_slots[key] == third_slots[key] or \
                    (best_slots[key] != second_slots[key] and best_slots[key] != third_slots[key] and
                     second_slots[key] != third_slots[key]):
                if isinstance(best_slots[key], set):
                    slot_dic[key] = list(best_slots[key])
                else:
                    slot_dic[key] = best_slots[key]
            elif second_slots[key] == third_slots[key]:
                if isinstance(second_slots[key], set):
                    slot_dic[key] = list(second_slots[key])
                else:
                    slot_dic[key] = second_slots[key]
        elif key in best_slots.keys() and key in second_slots.keys():  # 只有best 和 second 有
            if isinstance(best_slots[key], list) and isinstance(second_slots[key], list):
                best_slots[key] = set(best_slots[key])
                second_slots[key] = set(second_slots[key])

            if isinstance(best_slots[key], set):
                slot_dic[key] = list(best_slots[key])
            else:
                slot_dic[key] = best_slots[key]
        elif key in best_slots.keys() and key in third_slots.keys():
            if isinstance(best_slots[key], list) and isinstance(third_slots[key], list):
                best_slots[key] = set(best_slots[key])
                third_slots[key] = set(third_slots[key])
            if isinstance(best_slots[key], set):
                slot_dic[key] = list(best_slots[key])
            else:
                slot_dic[key] = best_slots[key]
        elif key in second_slots.keys() and key in third_slots.keys():
            if isinstance(second_slots[key], list) and isinstance(third_slots[key], list):
                second_slots[key] = set(second_slots[key])
                third_slots[key] = set(third_slots[key])
            if isinstance(second_slots[key], set):
                slot_dic[key] = list(second_slots[key])
            else:
                slot_dic[key] = second_slots[key]
        elif key in best_slots.keys():
            slot_dic[key] = best_slots[key]
        elif key in second_slots.keys():
            second_key.append(key)
        elif key in third_slots.keys():
            third_key.append(key)

    for key in second_key:
        if second_slots[key] not in slot_dic.values():
            slot_dic[key] = second_slots[key]
    for key in third_key:
        if third_slots[key] not in slot_dic.values():
            slot_dic[key] = third_slots[key]

    tmp['intent'] = intent
    tmp['slots'] = slot_dic
    res[idx] = tmp

print(res)
with open('../result/InteractModel/1/new_B/final_B_2.json', 'w', encoding='utf-8') as fp:
    json.dump(res, fp, ensure_ascii=False)
