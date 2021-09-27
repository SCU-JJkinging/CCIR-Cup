#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/14 19:58
# @Author  : JJkinging
# @File    : post_process.py
import json


def process(source_path, target_path):
    '''
    该函数作用：对模型预测结果进行纠正，即把不属于某一类intent的槽值删除，举例来说：
    比如我在一条测试数据中预测出其intent = FilmTele-Play, 然后其槽值预测中出现了"notes"这个槽标签，
    这与我之前统计的哪些槽标签只出现在哪些意图中不符合（即训练数据中FileTele-Play这个意图不可能出现"notes"这个槽标签），
    所以该函数就把"notes"这个槽位和槽值删除掉。
    :param source_path:
    :param target_path:
    :return:
    '''
    with open(source_path, 'r', encoding='utf-8') as fp:
        ori_data = json.load(fp)

    with open('../../user_data/common_data/intent_slot_mapping.json', 'r', encoding='utf-8') as fp:
        intent_slot_mapping = json.load(fp)

    mapping_keys = list(intent_slot_mapping.keys())
    for filename, single_data in ori_data.items():
        intent = single_data['intent']
        slots = single_data['slots']
        slot_keys = list(single_data['slots'].keys())
        for item in mapping_keys:
            if intent == item:
                all_tag = intent_slot_mapping[item]  # ["name", "tag", "artist", "region", "play_setting", "age"]
                for key in slot_keys:  # 检查slots结果的每一个槽位是否合理
                    if key not in all_tag:
                        ori_data[filename]['slots'].pop(key)  # 如果不合理，则删除该条槽

        for key, values in slots.items():
            if isinstance(values, list):
                tmp = list(set(values))
                if len(tmp) == 1:
                    tmp = tmp[0]
                ori_data[filename]['slots'][key] = tmp
    with open(target_path, 'w', encoding='utf-8') as fp:
        json.dump(ori_data, fp, ensure_ascii=False)
