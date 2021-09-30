#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/15 21:08
# @Author  : JJkinging
# @File    : extend_small_sample.py
'''数据增强: 扩充Audio-Play小样本数据集'''
import json
import random
random.seed(1000)

with open('../../extend_data/same_intent/ori_data/Audio-Play.json', 'r', encoding='utf-8') as fp:
    ori_data = json.load(fp)

with open('../../small_sample/audio_entity_dic.json', 'r', encoding='utf-8') as fp:
    audio_dic = json.load(fp)

res = {}
idx = 0
for filename, single_data in ori_data.items():
    text = single_data['text']
    slots = single_data['slots']
    for _ in range(20):  # 每条扩充20条数据
        tmp_dic = {}
        slot_dic = {}
        length = len(str(idx))
        o_num = 5 - length
        new_filename = 'NLU' + '0'*o_num + str(idx)
        idx += 1

        new_text = text
        for prefix in audio_dic['prefix']:
            if text.find(prefix) != -1:
                ran_i = random.randint(0, len(audio_dic['prefix'])-1)
                new_text = text.replace(prefix, audio_dic['prefix'][ran_i])  # 随机找一个prefix替换
                break
            else:
                ran_i = random.randint(0, len(audio_dic['prefix'])-1)
                new_text = audio_dic['prefix'][ran_i] + text
        if 'name' in slots.keys():
            ran_i = random.randint(0, len(audio_dic['name'])-1)
            new_text = new_text.replace(slots['name'], audio_dic['name'][ran_i])
            slot_dic['name'] = audio_dic['name'][ran_i]  # 随机替换name
        if 'artist' in slots.keys():
            ran_i = random.randint(0, len(audio_dic['artist']) - 1)
            if isinstance(slots['artist'], list):
                tmp_list = []
                for art in slots['artist']:
                    ran_j = random.randint(0, len(audio_dic['artist']) - 1)
                    new_text = new_text.replace(art, audio_dic['artist'][ran_j])
                    tmp_list.append(audio_dic['artist'][ran_j])
                slot_dic['artist'] = tmp_list
            else:
                new_text = new_text.replace(slots['artist'], audio_dic['artist'][ran_i])
                slot_dic['artist'] = audio_dic['artist'][ran_i]  # 随机替换artist
        if 'play_setting' in slots.keys():
            ran_i = random.randint(0, len(audio_dic['play_setting']) - 1)
            if isinstance(slots['play_setting'], list):
                tmp_list = []
                for set1 in slots['play_setting']:
                    ran_j = random.randint(0, len(audio_dic['play_setting']) - 1)
                    if set1 in audio_dic['play_setting']:
                        new_text = new_text.replace(set1, audio_dic['play_setting'][ran_j])
                        tmp_list.append(audio_dic['play_setting'][ran_j])
                    else:
                        tmp_list.append(set1)
                slot_dic['play_setting'] = tmp_list
            else:
                if slots['play_setting'] in audio_dic['play_setting']:
                    new_text = new_text.replace(slots['play_setting'], audio_dic['play_setting'][ran_i])
                    slot_dic['play_setting'] = audio_dic['play_setting'][ran_i]  # 随机替换play_setting
        if 'language' in slots.keys():
            ran_i = random.randint(0, len(audio_dic['language']) - 1)
            if slots['language'] == "俄语" and audio_dic['language'][ran_i] != "俄语":
                new_text = new_text.replace(slots['language'], audio_dic['language'][ran_i][:-1]+'文')
                slot_dic['language'] = audio_dic['language'][ran_i]  # 随机替换language
            elif slots['language'] == "俄语" and audio_dic['language'][ran_i] == "俄语":
                slot_dic['language'] = audio_dic['language'][ran_i]  # 随机替换language
            elif slots['language'] == "华语" and audio_dic['language'][ran_i] != "华语":
                new_text = new_text.replace("中文", audio_dic['language'][ran_i][:-1] + '文')
                slot_dic['language'] = audio_dic['language'][ran_i]  # 随机替换language
            elif slots['language'] == "华语" and audio_dic['language'][ran_i] == "华语":
                slot_dic['language'] = audio_dic['language'][ran_i]  # 随机替换language
            else:
                new_text = new_text.replace(slots['language'][:-1]+'文', audio_dic['language'][ran_i][:-1] + '文')
                slot_dic['language'] = audio_dic['language'][ran_i]  # 随机替换language
        if 'tag' in slots.keys():
            ran_i = random.randint(0, len(audio_dic['tag']) - 1)
            new_text = new_text.replace(slots['tag'], audio_dic['tag'][ran_i])
            slot_dic['tag'] = audio_dic['tag'][ran_i]  # 随机替换language
        tmp_dic['text'] = new_text
        tmp_dic['intent'] = "Audio-Play"
        tmp_dic['slots'] = slot_dic
        res[new_filename] = tmp_dic


print(res)
print(len(res))

with open('../../extend_data/data/extend_audio.json', 'w', encoding='utf-8') as fp:
    json.dump(res, fp, ensure_ascii=False)

# 手动替换 华语|华文 ——> 中文   西班牙文 ——> 西班牙语
