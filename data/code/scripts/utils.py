#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/8 13:09
# @Author  : JJkinging
# @File    : utils.py
import time
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from seqeval.metrics import precision_score as slot_precision_score, recall_score as slot_recall_score, \
    f1_score as slot_F1_score
from data.code.scripts.config_jointBert import Config


def load_vocab(vocab_file):
    '''construct word2id'''
    vocab = {}
    index = 0
    with open(vocab_file, 'r', encoding='utf-8') as fp:
        while True:
            token = fp.readline()
            if not token:
                break
            token = token.strip()  # 删除空白符
            vocab[token] = index
            index += 1
    return vocab


def collate_to_max_length(batch):
    # input_ids, slot_ids, input_mask, intent_id, slot_none_id
    batch_size = len(batch)
    input_ids_list = []
    slot_ids_list = []
    input_mask_list = []
    intent_ids_list = []
    slot_none_ids_list = []
    for single_data in batch:
        input_ids_list.append(single_data[0])
        slot_ids_list.append(single_data[1])
        input_mask_list.append(single_data[2])
        intent_ids_list.append(single_data[3])
        slot_none_ids_list.append(single_data[4])

    max_length = max([len(item) for item in input_ids_list])

    output = [torch.full([batch_size, max_length],
                         fill_value=0,
                         dtype=torch.long),
              torch.full([batch_size, max_length],
                         fill_value=0,
                         dtype=torch.long),
              torch.full([batch_size, max_length],
                         fill_value=0,
                         dtype=torch.long)
              ]

    for i in range(batch_size):
        output[0][i][0:len(input_ids_list[i])] = torch.LongTensor(input_ids_list[i])
        output[1][i][0:len(slot_ids_list[i])] = torch.LongTensor(slot_ids_list[i])
        output[2][i][0:len(input_mask_list[i])] = torch.LongTensor(input_mask_list[i])

    intent_ids_list = torch.LongTensor(intent_ids_list)
    output.append(intent_ids_list)

    slot_none_ids = torch.zeros([batch_size, 29], dtype=torch.long)
    for i, slot_none_id in enumerate(slot_none_ids_list):
        for idx in slot_none_id:
            slot_none_ids[i][idx] = 1

    output.append(slot_none_ids)

    return output  # (input_ids, slot_ids, input_mask, intent_id, slot_none_id)


def train(model,
          dataloader,
          optimizer,
          I_criterion,
          N_criterion,
          max_gradient_norm):
    model.train()
    # device = model.module.device
    device = model.device
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    preds_mounts = 0

    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, batch in enumerate(tqdm_batch_iterator):
        batch_start = time.time()
        input_ids, slot_ids, input_mask, intent_id, slot_none_id = batch

        input_id = input_ids.to(device)
        slot_ids = slot_ids.to(device)
        input_mask = input_mask.byte().to(device)
        intent_id = intent_id.to(device)
        slot_none_id = slot_none_id.to(device)

        optimizer.zero_grad()
        intent_logits, slot_none_logits, slot_logits = model(input_id, input_mask)
        # intent_loss = CE_criterion(intent_logits, intent_id)
        intent_loss = I_criterion(intent_logits, intent_id)
        # slot_none_loss = BCE_criterion(slot_none_logits, slot_none_id.float())
        slot_none_loss = N_criterion(slot_none_logits, slot_none_id.float())
        slot_loss = model.slot_loss(slot_logits, slot_ids, input_mask)
        loss = intent_loss + slot_none_loss + slot_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)

        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()

        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
            .format(batch_time_avg / (batch_index + 1),
                    running_loss / (batch_index + 1))
        tqdm_batch_iterator.set_description(description)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)

    return epoch_time, epoch_loss


def valid(model,
          dataloader,
          I_criterion,
          N_criterion,):
    config = Config()
    model.eval()
    # device = model.module.device
    device = model.device
    epoch_start = time.time()
    running_loss = 0.0
    preds_mounts = 0
    sen_counts = 0

    intent_true = []
    intent_pred = []
    slot_pre_output = []
    slot_true_output = []
    slotNone_pre_output = []
    slotNone_true_output = []

    slot_dict = load_vocab(config.slot_label)
    id2slot = {value: key for key, value in slot_dict.items()}

    with torch.no_grad():
        tqdm_batch_iterator = tqdm(dataloader)
        for _, batch in enumerate(tqdm_batch_iterator):
            input_ids, slot_ids, input_mask, intent_id, slot_none_id = batch
            batch_size = len(input_ids)
            input_ids = input_ids.to(device)
            slot_ids = slot_ids.to(device)
            input_mask = input_mask.byte().to(device)
            intent_id = intent_id.to(device)
            slot_none_id = slot_none_id.to(device)

            real_length = torch.sum(input_mask, dim=1)
            tmp = []
            i = 0
            for line in slot_ids.cpu().numpy().tolist():
                line = [id2slot[idx] for idx in line[1: real_length[i]-1]]
                tmp.append(line)
                i += 1

            slot_true_output.extend(tmp)

            intent_logits, slot_none_logits, slot_logits = model(input_ids, input_mask)
            # intent_loss = CE_criterion(intent_logits, intent_id)
            intent_loss = I_criterion(intent_logits, intent_id)

            # slot_none_loss = BCE_criterion(slot_none_logits, slot_none_id.float())
            slot_none_loss = N_criterion(slot_none_logits, slot_none_id.float())
            slot_none_probs = torch.sigmoid(slot_none_logits)
            slot_none_probs = slot_none_probs > 0.5
            slot_none_probs = slot_none_probs.cpu().numpy()
            slot_none_probs = slot_none_probs.astype(int)
            slot_none_probs = slot_none_probs.tolist()
            slotNone_pre_output.extend(slot_none_probs)

            slot_none_id = slot_none_id.cpu().numpy().tolist()
            slotNone_true_output.extend(slot_none_id)

            slot_loss = model.slot_loss(slot_logits, slot_ids, input_mask)
            out_path = model.slot_predict(slot_logits, input_mask, id2slot)
            out_path = [[id2slot[idx] for idx in one_data[1:-1]] for one_data in out_path]  # 去掉'[START]'和'[EOS]'标记
            slot_pre_output.extend(out_path)
            loss = intent_loss + slot_none_loss + slot_loss
            running_loss += loss.item()

            # intent acc
            intent_probs = torch.softmax(intent_logits, dim=-1)
            predict_labels = torch.argmax(intent_probs, dim=-1)
            correct_preds = (predict_labels == intent_id).sum().item()
            preds_mounts += correct_preds

            predict_labels = predict_labels.cpu().numpy().tolist()
            intent_pred.extend(predict_labels)
            intent_id = intent_id.cpu().numpy().tolist()
            intent_true.extend(intent_id)

    # 计算slotNone classification 准确率、召回率、F1值
    slotNone_micro_acc = precision_score(slotNone_true_output, slotNone_pre_output, average='micro')
    slotNone_micro_recall = recall_score(slotNone_true_output, slotNone_pre_output, average='micro')
    slotNone_micro_f1 = f1_score(slotNone_true_output, slotNone_pre_output, average='micro')

    # 计算slot filling 准确率、召回率、F1值
    slot_acc = slot_precision_score(slot_true_output, slot_pre_output)
    slot_recall = slot_recall_score(slot_true_output, slot_pre_output)
    slot_f1 = slot_F1_score(slot_true_output, slot_pre_output)

    # 计算整句正确率
    for i in range(len(intent_true)):
        if intent_true[i] == intent_pred[i] and slot_true_output[i] == slot_pre_output[i] and \
                slotNone_pre_output[i] == slotNone_true_output[i]:
            sen_counts += 1
    # fp = open('../error_list1', 'w', encoding='utf-8')
    # for i in range(len(intent_true)):
    #     if intent_true[i] != intent_pred[i] or slot_true_output[i] != slot_pre_output[i] or \
    #     slotNone_true_output[i] != slotNone_pre_output[i]:
    #         fp.write(str(i)+'\n')

    slot_none = (slotNone_micro_acc, slotNone_micro_recall, slotNone_micro_f1)
    slot = (slot_acc, slot_recall, slot_f1)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    intent_accuracy = preds_mounts / len(dataloader.dataset)
    sen_accuracy = sen_counts / len(dataloader.dataset)

    return epoch_time, epoch_loss, intent_accuracy, slot_none, slot, sen_accuracy
