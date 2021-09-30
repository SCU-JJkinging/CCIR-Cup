#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/19 10:39
# @Author  : JJkinging
# @File    : rectify.py
import json


def rectify(source_path, target_path):
    with open(source_path, 'r', encoding='utf-8') as fp:
        ori_data = json.load(fp)

    for filename, single_data in ori_data.items():
        text = single_data['text']
        slots = single_data['slots']
        intent = single_data['intent']
        if intent == 'Alarm-Update':
            if 'datetime_time' in slots.keys():
                idx_1 = text.find(':')
                idx_2 = text[idx_1+1:].find(':')

                if idx_1 != -1:
                    start_idx = idx_1
                    end_idx = idx_1
                    for i in range(1, 3):  # 找到第一个时间的起始index
                        if text[start_idx - 1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                            start_idx -= 1
                        if text[end_idx + 1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                            end_idx += 1
                    if idx_2 != -1:  # 说明有两个时间
                        start_idx_2 = idx_2 + idx_1 + 1  # 修正一下
                        end_idx_2 = start_idx_2
                        for i in range(1, 3):  # 找到第二个时间的起始index
                            if text[start_idx_2 - 1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                                start_idx_2 -= 1
                            if text[end_idx_2 + 1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                                end_idx_2 += 1
                        for item in slots['datetime_time']:
                            if item in text[start_idx: end_idx+1]:
                                slots['datetime_time'].remove(item)
                                slots['datetime_time'].append(text[start_idx: end_idx+1])
                            elif item in text[start_idx_2: end_idx_2+1]:
                                slots['datetime_time'].remove(item)
                                slots['datetime_time'].append(text[start_idx_2: end_idx_2 + 1])
                    else:
                        if isinstance(slots['datetime_time'], list):
                            for item in slots['datetime_time']:
                                if item in text[start_idx: end_idx + 1]:
                                    slots['datetime_time'].remove(item)
                                    slots['datetime_time'].append(text[start_idx: end_idx+1])
                        else:
                            index = slots['datetime_time'].find(':')
                            if index != -1:
                                slots['datetime_time'] = slots['datetime_time'][:index] + text[idx_1: end_idx+1]
                            else:
                                slots['datetime_time'] = slots['datetime_time'] + text[idx_1: end_idx + 1]

    with open(target_path, 'w', encoding='utf-8') as fp:
        json.dump(ori_data, fp, ensure_ascii=False)


if __name__ == "__main__":
    # source = '../small_sample/data/train_8_with_other.json'
    # target = '../small_sample/data/train_8_with_other_clear.json'
    source = '../small_sample/new_B/train_final.json'
    target = '../small_sample/new_B/train_final_clear.json'
    rectify(source, target)

'''手动改'datetime_time': []'''

'''先删除
NLU12276
"NLU11929": {
    "text": "能不能建一个今天晚上6:30~7:00提醒我钉钉直播会议",
    "intent": "Alarm-Update",
    "slots": {
      "datetime_date": "今天",
      "datetime_time": "晚上6:30~7:00",
      "notes": "钉钉直播会议"
    }
  },
再添加
'''
