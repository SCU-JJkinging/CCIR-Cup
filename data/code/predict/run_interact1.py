#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/9 15:22
# @Author  : JJkinging
# @File    : run_predict.py
import warnings

import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.code.scripts.config_Interact1 import Config
from data.code.predict.test_utils import load_reverse_vocab, load_vocab, collate_to_max_length
from data.code.predict.test_dataset import CCFDataset
from data.code.model.InteractModel_1 import InteractModel
from transformers import AutoTokenizer
from data.code.predict.post_process import process


def run_test(model, dataloader, slot_dict):

    model.eval()
    device = model.device
    intent_probs_list = []

    intent_pred = []
    slotNone_pred_output = []
    slot_pre_output = []
    input_ids_list = []

    id2slot = {value: key for key, value in slot_dict.items()}
    with torch.no_grad():
        tqdm_batch_iterator = tqdm(dataloader)
        for _, batch in enumerate(tqdm_batch_iterator):
            input_ids, input_mask = batch

            input_ids = input_ids.to(device)
            input_mask = input_mask.byte().to(device)
            real_length = torch.sum(input_mask, dim=1)

            intent_logits, slot_none_logits, slot_logits = model(input_ids, input_mask)

            # intent
            intent_probs = torch.softmax(intent_logits, dim=-1)
            predict_labels = torch.argmax(intent_probs, dim=-1)
            predict_labels = predict_labels.cpu().numpy().tolist()
            intent_pred.extend(predict_labels)

            # slot_none
            slot_none_probs = torch.sigmoid(slot_none_logits)
            slot_none_probs = slot_none_probs > 0.5
            slot_none_probs = slot_none_probs.cpu().numpy()
            slot_none_probs = slot_none_probs.astype(int)
            slot_none_probs = slot_none_probs.tolist()

            for slot_none_id in slot_none_probs:  # 遍历这个batch的slot_none_id
                tmp = []
                if 1 in slot_none_id:
                    for idx, none_id in enumerate(slot_none_id):
                        if none_id == 1:
                            tmp.append(idx)
                else:
                    tmp.append(28)  # 28在slot_none_vocab中表示none
                slotNone_pred_output.append(tmp)  # [[4, 5], [], [], ...]

            # slot
            out_path = model.slot_predict(slot_logits, input_mask, id2slot)
            out_path = [[id2slot[idx] for idx in one_data] for one_data in out_path]  # 不去掉'[START]'和'[EOS]'标记
            slot_pre_output.extend(out_path)

            # input_ids
            input_ids = input_ids.cpu().numpy().tolist()
            input_ids_list.extend(input_ids)

    return intent_pred, slotNone_pred_output, slot_pre_output, input_ids_list


def get_slot(tokenizer, tmp_dict, input_ids, ori_sen, start, end, key):
    value = tokenizer.decode(input_ids[start:end + 1])
    value = ''.join(value.split(' ')).replace('[UNK]', '$')
    s = -1
    e = -1
    ori_sen_lower = ori_sen.lower()  # 把其中的英文字母全变为小写
    is_unk = False
    if len(value) > 0:
        if value[0] == '$':  # 如果第一个就是[UNK]
            value = value[1:]  # 查找时先忽略，最后再把开始位置往前移一位
            is_unk = True
    i = 0
    j = 0
    while i < len(ori_sen_lower) and j < len(value):
        if ori_sen_lower[i] == value[j]:
            if s == -1:
                s = i
                e = i
            else:
                e += 1
            i += 1
            j += 1
        elif ori_sen_lower[i] == ' ':
            e += 1
            i += 1
        elif value[j] == '$':
            e += 1
            i += 1
            j += 1
        elif ori_sen_lower[i] != value[j]:
            i -= j - 1
            j = 0
            s = -1
            e = -1
    if is_unk:
        s = s-1
    final_value = ori_sen[s:e + 1]
    if key in tmp_dict.keys():
        if tmp_dict[key] != '' and not isinstance(tmp_dict[key], list):  # 如果该key已经有值，且不为list
            tmp_list = [tmp_dict[key], final_value]
            tmp_dict[key] = tmp_list
        elif tmp_dict[key] != '' and isinstance(tmp_dict[key], list):
            tmp_dict[key].append(final_value)
    else:
        tmp_dict[key] = final_value
    return tmp_dict


def is_nest(slot_pre):
    '''
    判断该条数据的 slot filling 是否存在嵌套
    :param pred_slot: 例如：['O', 'O', 'O', 'B-age', 'I-age', 'O', 'O', 'O']
    :return: 存在返回True, 否则返回False
    '''
    for j, cur_tag in enumerate(slot_pre):
        if cur_tag.startswith('I-'):
            cur_tag_name = cur_tag[2:]
            if j < len(slot_pre)-1:
                if slot_pre[j+1].startswith('I-'):  # 如果下一个也是'I-'开头
                    post_tag_name = slot_pre[j+1][2:]
                    if cur_tag_name != post_tag_name:  # 但二者不等
                        return True
    return False


def process_nest(tokenizer, input_ids, slot_pre, ori_sen):  # 处理嵌套
    '''
    :param input_ids: 原句的id形式（wordpiece过）
    :param slot_pre: 该句的预测slot
    :param ori_sen: 原句（字符）
    :return:
    '''
    start_outer = -1
    end_outer = -1
    tmp_dict = {}
    pre_end = -1
    for i, tag in enumerate(slot_pre):
        if i <= pre_end:
            continue
        if tag.startswith('B-') and start_outer == -1:  # 第一个'B-'
            start_outer = i  # 临时起始位置
            end_outer = i
            key_outer = tag[2:]

        elif tag.startswith('I-') and tag[2:] == slot_pre[i-1][2:]:  # 'I-' 且标签与前一个slot相同
            end_outer = i
            if i == len(slot_pre)-1:  # 到了最后
                tmp_dict = get_slot(tokenizer, tmp_dict, input_ids, ori_sen, start_outer, end_outer, key_outer)

        elif tag.startswith('O') and start_outer == -1:
            continue
        elif tag.startswith('O') and start_outer != -1:
            tmp_dict = get_slot(tokenizer, tmp_dict, input_ids, ori_sen, start_outer, end_outer, key_outer)
            start_outer = -1
            end_outer = -1
        # 第一种嵌套：B-region I-region I-name I-name
        elif tag.startswith('I-') and tag[2:] != slot_pre[i-1][2:]:  # 'I-' 且标签与前一个slot不同
            start_inner = start_outer
            end_inner = end_outer
            key_inner = key_outer
            # 处理内层
            tmp_dict = get_slot(tokenizer, tmp_dict, input_ids, ori_sen, start_inner, end_inner, key_inner)
            # start_outer不变
            end_outer = i  # end_outer为当前slot下标
            key_outer = slot_pre[i][2:]  # 需修改key_outer
        # 第二种嵌套：B-name I-name B-datetime_date I-datetime_date I-name B-region I-region I-name I-name
        elif tag.startswith('B-'):
            flag = False
            pre_start = start_outer
            pre_end = end_outer
            pre_key = key_outer
            start_outer = i
            end_outer = i
            key_outer = slot_pre[i][2:]
            # ************************
            for j in range(i, len(slot_pre)):
                if slot_pre[j] != 'O' and slot_pre[j][2:] == pre_key:
                    flag = True
                    pre_end = j
                    slot_pre[j] = 'O'  # 置为'O'
            # *************************
            if flag:  # 上一个slot是嵌套
                tmp_dict = get_slot(tokenizer, tmp_dict, input_ids, ori_sen, pre_start, pre_end, pre_key) #先处理外层
                # 以后遇到完整slot的就加入(假设只有二级嵌套)
                for k in range(i+1, pre_end+1):
                    if slot_pre[k] != 'O':
                        if slot_pre[k].startswith('I-'):
                            end_outer = k
                        elif slot_pre[k].startswith('B-') and start_outer == -1:
                            start_outer = k
                            end_outer = k
                            key_outer = slot_pre[k][2:]
                        elif slot_pre[k].startswith('B-') and start_outer != -1:
                            tmp_dict = get_slot(tokenizer, tmp_dict, input_ids, ori_sen, start_outer, end_outer,
                                                key_outer)  # 处理内层
                            start_outer = k
                            end_outer = k
                            key_outer = slot_pre[k][2:]
                    elif slot_pre[k] == 'O' and start_outer != -1:
                        tmp_dict = get_slot(tokenizer, tmp_dict, input_ids, ori_sen, start_outer, end_outer,
                                            key_outer)  # 处理内层
                        start_outer = -1
                        end_outer = -1
                        key_outer = None
                    elif slot_pre[k] == 'O' and start_outer == -1:
                        continue
            else:  # 上一个不是嵌套
                tmp_dict = get_slot(tokenizer, tmp_dict, input_ids, ori_sen, pre_start, pre_end, pre_key) #先处理上一个非嵌套

    return tmp_dict


def find_slot(tokenizer, input_ids_list, slot_pre_output, ori_sen_list):
    fp = open('nest.txt', 'w+', encoding='utf-8')
    slot_list = []
    count = 0
    # ddd = 0
    for i, slot_ids in enumerate(slot_pre_output):  # 遍历每条数据
        # if ddd < 1111:
        #     ddd += 1
        #     continue
        tmp_dict = {}
        start = 0
        end = 0
        if is_nest(slot_ids[1:-1]):  # 如果确实存在嵌套行为
            tmp_dict = process_nest(tokenizer, input_ids_list[i][1:-1], slot_pre_output[i][1:-1], ori_sen_list[i])
            slot_list.append(tmp_dict)
            fp.write(str(ori_sen_list[i])+'\t')
            fp.write(str(tmp_dict)+'\n')
            fp.write(' '.join(slot_pre_output[i][1:-1])+'\n')
            count += 1
        else:
            for j, slot in enumerate(slot_ids):  # 遍历每个字
                if slot != 'O' and slot != '[START]' and slot != '[EOS]':
                    if slot.startswith('B-') and start == 0:
                        start = j  # 槽值起始位置
                        end = j
                        key = slot[2:]
                        if j == len(slot_ids)-2:  # # 如果'B-'是最后一个字符(即倒数第二个，真倒数第一是EOS)
                            tmp_dict = get_slot(tokenizer, tmp_dict, input_ids_list[i], ori_sen_list[i], start, end, key)
                            break
                    elif slot.startswith('B-') and start != 0:  # 说明上一个是槽
                        # tokenizer, tmp_dict, input_ids, ori_sen, start, end, key
                        tmp_dict = get_slot(tokenizer, tmp_dict, input_ids_list[i], ori_sen_list[i], start, end, key)  # 将槽值写入
                        start = j
                        end = j
                        key = slot[2:]
                    else:  # 'I-'开头
                        end += 1
                        if j == len(slot_ids)-2:  # 如果'I-'是最后一个字符(即倒数第二个，真倒数第一是EOS)
                            tmp_dict = get_slot(tokenizer, tmp_dict, input_ids_list[i], ori_sen_list[i], start, end, key)
                            break
                else:
                    if end == 0:  # 说明没找到槽
                        continue
                    else:
                        if slot != '[EOS]':
                            tmp_dict = get_slot(tokenizer, tmp_dict, input_ids_list[i], ori_sen_list[i], start, end, key)
                            start = 0
                            end = 0
            slot_list.append(tmp_dict)
    print(str(count)+'——interact1 推理完成！')
    fp.close()
    return slot_list


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = Config()
    device = torch.device(config.cuda if torch.cuda.is_available() else "cpu")
    with open('../../user_data/common_data/region_dic.json', 'r', encoding='utf-8') as fp:
        region_dict = json.load(fp)
    tokenizer = AutoTokenizer.from_pretrained('../../user_data/pretrained_model/ernie')
    print('interact1 开始推理！')
    vocab = load_vocab('../../user_data/pretrained_model/ernie/vocab.txt')
    id2intent = load_reverse_vocab('../../user_data/common_data/intent_label.txt')
    id2slotNone = load_reverse_vocab('../../user_data/common_data/slot_none_vocab.txt')
    id2slot = load_reverse_vocab('../../user_data/common_data/slot_label.txt')

    intent_dict = load_vocab('../../user_data/common_data/intent_label.txt')
    slot_none_dict = load_vocab('../../user_data/common_data/slot_none_vocab.txt')
    slot_dict = load_vocab('../../user_data/common_data/slot_label.txt')

    intent_tagset_size = len(intent_dict)
    slot_none_tag_size = len(slot_none_dict)
    slot_tag_size = len(slot_dict)
    test_filename = '../../user_data/test_data/test_seq_in_B.txt'
    test_dataset = CCFDataset(test_filename, vocab, intent_dict, slot_none_dict, slot_dict, config.max_length)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                             collate_fn=collate_to_max_length)

    model = InteractModel('../../user_data/pretrained_model/ernie',
                          config.bert_hidden_size,
                          intent_tagset_size,
                          slot_none_tag_size,
                          slot_tag_size,
                          device).to(device)
    checkpoint = torch.load('../../user_data/output_model/InteractModel_1/Interact1_model_best.pth.tar')
    model.load_state_dict(checkpoint["model"])

    # -------------------- Testing ------------------- #
    print("\n",
          20 * "=",
          "Test model on device: {}".format(device),
          20 * "=")

    intent_pred, slotNone_pred_output, slot_pre_output, input_ids_list = run_test(model, test_loader, slot_dict)

    ori_sen_list = []
    with open('../../user_data/test_data/test_B_final_text.json', 'r', encoding='utf-8') as fp:
        raw_data = json.load(fp)
    for filename, single_data in raw_data.items():
        text = single_data['text']
        ori_sen_list.append(text)
    slot_list = find_slot(tokenizer, input_ids_list, slot_pre_output, ori_sen_list)

    # 用region_dict对slot_list中需要替换的词进行替换 (这一步不能忽略！！！！)
    for i, slot_dict in enumerate(slot_list):
        for slot, slot_value in slot_dict.items():
            for region_key, region_list in region_dict.items():
                if slot_value in region_list:
                    slot_list[i][slot] = region_key
    res = {}
    for i in range(len(intent_pred)):
        big_tmp = {}
        slot_tmp_dict = {}
        # intent
        intent = id2intent[intent_pred[i]]
        # slot_none
        for slot_none_id in slotNone_pred_output[i]:
            if 0 <= slot_none_id <= 9:
                slot_tmp_dict['command'] = id2slotNone[slot_none_id]
            elif 10 <= slot_none_id <= 18:
                slot_tmp_dict['index'] = id2slotNone[slot_none_id]
            elif 19 <= slot_none_id <= 22:
                slot_tmp_dict['play_mode'] = id2slotNone[slot_none_id]
            elif 23 <= slot_none_id <= 27:
                slot_tmp_dict['query_type'] = id2slotNone[slot_none_id]
        # slot
        slot_tmp_dict.update(slot_list[i])

        big_tmp['intent'] = intent
        big_tmp['slots'] = slot_tmp_dict

        length = len(str(i))
        o_num = 5 - length
        index = 'NLU' + '0'*o_num + str(i)
        res[index] = big_tmp

    with open('../../user_data/tmp_result/result_interact_1.json', 'w', encoding='utf-8') as fp:
        json.dump(res, fp, ensure_ascii=False)

    source_path = '../../user_data/tmp_result/result_interact_1.json'
    target_path = '../../user_data/tmp_result/result_interact_1_post.json'
    process(source_path, target_path)
