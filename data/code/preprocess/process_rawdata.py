#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/30 23:32
# @Author  : JJkinging
# @File    : process_rawdata.py
import json
from transformers import AutoTokenizer


def find_idx(text, value, dic, mode):
    '''
    返回value在text中的起始位置
    :param text: list
    :param value: list
    :param dic: 同义词词表, 只有mode=='mohu'时才用到
    :param mode: 如果mode=='mohu',则表示是模糊搜索
    :return:
    '''

    if mode == 'mohu':
        value = ''.join(value)  # 变成字符串
        if value not in dic.keys():
            return -1
        candidate = dic[value]
        flag = False
        index = -1
        for ca_value in candidate:
            for idx, item in enumerate(text):
                if item == ca_value[0]:  # 匹配第一个字符
                    index = idx
                    for i in range(len(ca_value)):
                        if text[idx + i] == ca_value[i]:
                            flag = True
                            if i == len(ca_value) - 1:
                                return index
                        else:
                            index = -1
                            flag = False
                            break
    else:
        flag = False
        index = -1
        for idx, item in enumerate(text):
            if item == value[0]:  # 匹配第一个
                index = idx
                for i in range(len(value)):
                    if text[idx + i] == value[i]:
                        flag = True
                        if i == len(value) - 1:
                            return index
                    else:
                        index = -1
                        flag = False
                        break
    if flag:
        return index
    else:
        return -1


def load_reverse_vocab(label_file):
    '''construct id2word'''
    vocab = {}
    index = 0
    with open(label_file, 'r', encoding='utf-8') as fp:
        while True:
            token = fp.readline()
            if not token:
                break
            token = token.strip()  # 删除空白符
            vocab[index] = token
            index += 1
    return vocab


def fun(filename,
        seq_in_target_file,
        seq_out_target_file,
        slot_none_target_file,
        region_dict_file,
        vocab):
    # tokenizer = AutoTokenizer.from_pretrained('nghuyong/ernie-1.0')
    tokenizer = AutoTokenizer.from_pretrained('nghuyong/ernie-1.0')

    with open(filename, 'r', encoding='utf-8') as fp:
        raw_data = json.load(fp)

    with open(region_dict_file, 'r', encoding='utf-8') as fp:
        region_dic = json.load(fp)

    seq_in = open(seq_in_target_file, 'a', encoding='utf-8')
    seq_out = open(seq_out_target_file, 'a', encoding='utf-8')
    slot_none_out = open(slot_none_target_file, 'a', encoding='utf-8')
    label = open('../new_data/label', 'a', encoding='utf-8')
    query = open('../new_data/query', 'a', encoding='utf-8')
    command = open('../new_data/command', 'a', encoding='utf-8')
    play_mode = open('../new_data/play_mode', 'a', encoding='utf-8')
    index = open('../new_data/index', 'a', encoding='utf-8')

    intent_label = set()
    slot_label = set()
    command_type = set()
    index_type = set()
    play_mode_type = set()
    query_type = set()

    count = 0
    raw_data_tmp = raw_data.copy()
    for dataname, single_data in raw_data.items():
        flag = True
        text = single_data['text']
        if text[-1] == '。':  # 去掉末尾的句号
            text = text[:-1]
        text_id = tokenizer.encode(text)  # 包含1和2
        text = [vocab[idx] for idx in text_id[1:-1]]
        intent = single_data['intent']
        slots = single_data['slots']
        slot_tags_str = ('O ' * len(text)).strip(' ')
        slot_tags = [item for item in slot_tags_str]  # 初始化句子槽标签全为'O'
        # label.write(intent+'\n')
        # intent_label.add(intent)

        slot_none_str = ''

        for slot, value in slots.items():
            if slot == 'command':
                command_type.add(str(value))
                if isinstance(value, list):
                    for item in value:  # value是个列表，item是字符串槽值
                        slot_none_str += item + ' '
                else:
                    slot_none_str += value + ' '
                # command.write(str(value)+'\n')
            elif slot == 'index':
                index_type.add(str(value))
                if isinstance(value, list):
                    for item in value:  # value是个列表，item是字符串槽值
                        slot_none_str += item + ' '
                else:
                    slot_none_str += value + ' '
                # index.write(str(value)+'\n')
            elif slot == 'play_mode':
                play_mode_type.add(str(value))
                if isinstance(value, list):
                    for item in value:  # value是个列表，item是字符串槽值
                        slot_none_str += item + ' '
                else:
                    slot_none_str += value + ' '
                # play_mode.write(str(value)+'\n')
            elif slot == 'query_type':
                query_type.add(str(value))
                if isinstance(value, list):
                    for item in value:  # value是个列表，item是字符串槽值
                        slot_none_str += item + ' '
                else:
                    slot_none_str += value + ' '
                # query.write(str(value)+'\n')
            else:
                # query.write('None'+'\n')
                # command.write('None'+'\n')
                # play_mode.write('None'+'\n')
                # index.write('None'+'\n')
                slot_label.add(slot)
                if isinstance(value, list):  # 槽不止一个值 例如："artist": ["陈博","嘉洋"]
                    for v in value:
                        v = [vocab[idx] for idx in tokenizer.encode(v)[1:-1]]
                        start_idx = find_idx(text, v, region_dic, mode='normal')
                        if start_idx == -1:
                            start_idx = find_idx(text, value, region_dic, mode='mohu')  # 二次检测
                            if start_idx == -1:
                                flag = False
                                print("无此槽值:", slot, v, dataname)
                                raw_data_tmp.pop(dataname)
                                count += 1
                                break
                        tag_len = len(v)
                        if tag_len == 1:
                            slot_tags[start_idx * 2] = 'B-' + slot
                        else:
                            for i in range(tag_len):
                                if i == 0:
                                    slot_tags[(start_idx + i) * 2] = 'B-' + slot
                                else:
                                    slot_tags[(start_idx + i) * 2] = 'I-' + slot

                else:  # 槽值为单个字符串
                    value = [vocab[idx] for idx in tokenizer.encode(value)[1:-1]]
                    if not value:  # value 为空
                        break
                    start_idx = find_idx(text, value, region_dic, mode='normal')  # 槽标签起始位置
                    if start_idx == -1:  # 句子无此槽值时
                        start_idx = find_idx(text, value, region_dic, mode='mohu')  # 二次检测
                        if start_idx == -1:
                            flag = False
                            print("无此槽值:", slot, value, dataname)
                            raw_data_tmp.pop(dataname)
                            count += 1
                            break
                    tag_len = len(value)  # 槽标签长度
                    if tag_len == 1:
                        slot_tags[start_idx * 2] = 'B-' + slot
                    else:
                        for i in range(tag_len):
                            if i == 0:
                                slot_tags[(start_idx + i) * 2] = 'B-' + slot
                            else:
                                slot_tags[(start_idx + i) * 2] = 'I-' + slot

        if flag:
            if slot_none_str == '':  # slot_none_str为空，说明此句话无slot_none槽类型
                slot_none_out.write('None' + '\n')
            else:
                slot_none_out.write(slot_none_str + '\n')

            slot_tags = ''.join(slot_tags)  # 把slot_tags变为str类型
            seq_out.write(slot_tags + '\n')
            seq_in.write(' '.join(text) + '\n')

    intent_label = ' '.join(intent_label)
    slot_label = ' '.join(slot_label)

    # with open('../new_data/intent_label.txt', 'a', encoding='utf-8') as fp:
    #     fp.write(intent_label+'\n')
    # with open('../new_data/slot_label.txt', 'a', encoding='utf-8') as fp:
    #     fp.write(slot_label+'\n')
    # with open('../new_data/command_type.txt', 'a', encoding='utf-8') as fp:
    #     fp.write(str(command_type)+'\n')
    # with open('../new_data/index_type.txt', 'a', encoding='utf-8') as fp:
    #     fp.write(str(index_type)+'\n')
    # with open('../new_data/play_mode_type.txt', 'a', encoding='utf-8') as fp:
    #     fp.write(str(play_mode_type)+'\n')
    # with open('../new_data/query_type.txt', 'a', encoding='utf-8') as fp:
    #     fp.write(str(query_type)+'\n')

    seq_in.close()
    seq_out.close()
    slot_none_out.close()
    with open('../small_sample/new_B/train_final_clear_del.json', 'w', encoding='utf-8') as fp:
        json.dump(raw_data_tmp, fp, ensure_ascii=False)
    print('缺失数目:', count)


if __name__ == "__main__":
    vocab_file = '../pretrained_model/erine/vocab.txt'
    train_filename = '../small_sample/new_B/train_final_clear_sorted.json'
    dev_filename = '../extend_data/last_data/dev_sorted.json'
    test_filename = '../raw_data/test_B_final_text.json'
    seq_in_target_file = '../small_sample/new_B/train_seq_in.txt'
    seq_out_target_file = '../small_sample/new_B/train_seq_out.txt'
    slot_none_target_file = '../small_sample/new_B/train_slot_none.txt'
    region_dict_file = '../final_data/region_dic.json'
    vocab = load_reverse_vocab(vocab_file)
    fun(filename=train_filename,
        seq_in_target_file=seq_in_target_file,
        seq_out_target_file=seq_out_target_file,
        slot_none_target_file=slot_none_target_file,
        region_dict_file=region_dict_file,
        vocab=vocab)
    # text = ['下', '周', '六', '我', '爷', '爷', '让', '我', '去', '买', '茶', '叶', '，', '记', '得', '提', '醒', '我']
    # value = ['我', '爷', '爷', '让', '我', '去', '买', '茶', '叶']
    # idx = find_idx(text, value)
