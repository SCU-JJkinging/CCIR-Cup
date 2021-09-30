#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/8 16:10
# @Author  : JJkinging
# @File    : split_train_dev.py
import json


def split_train_dev(ratio=0.8):
    with open('../raw_data/train_slot_sorted.json', 'r', encoding='utf-8') as fp:
        raw_data = json.load(fp)

    length = len(raw_data)

    train_data = {}
    dev_data = {}

    count = 0
    for key, value in raw_data.items():
        if count < length*ratio:
            train_data[key] = value
        else:
            dev_data[key] = value
        count += 1

    # 写训练集
    with open('../raw_data/train_8.json', 'w', encoding='utf-8') as fp:
        json.dump(train_data, fp, ensure_ascii=False)
    # 写验证集
    with open('../raw_data/dev_2.json', 'w', encoding='utf-8') as fp:
        json.dump(dev_data, fp, ensure_ascii=False)


if __name__ == "__main__":
    split_train_dev(0.8)
