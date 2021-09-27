#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/8 14:41
# @Author  : JJkinging
# @File    : main.py

import os
import torch
import random
import numpy as np
from transformers import get_linear_schedule_with_warmup
from data.code.scripts.dataset import CCFDataset
from data.code.scripts.config_Interact1 import Config
from torch.utils.data import DataLoader
from data.code.model.InteractModel_1 import InteractModel
from data.code.scripts.utils import train, valid, collate_to_max_length, load_vocab


def torch_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def main():
    torch_seed(1000)
    # 设置GPU数目
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = Config()
    device = torch.device(config.cuda if torch.cuda.is_available() else "cpu")
    print('loading corpus')
    vocab = load_vocab(config.vocab_file)
    intent_dict = load_vocab(config.intent_label_file)
    slot_none_dict = load_vocab(config.slot_none_vocab)
    slot_dict = load_vocab(config.slot_label)
    intent_tagset_size = len(intent_dict)
    slot_none_tag_size = len(slot_none_dict)
    slot_tag_size = len(slot_dict)
    train_dataset = CCFDataset(config.train_file, config.train_intent_file, config.train_slot_filename,
                               config.train_slot_none_filename, vocab, intent_dict, slot_none_dict, slot_dict,
                               config.max_length)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                              collate_fn=collate_to_max_length)

    dev_dataset = CCFDataset(config.dev_file, config.dev_intent_file, config.dev_slot_filename,
                             config.dev_slot_none_filename, vocab, intent_dict, slot_none_dict, slot_dict,
                             config.max_length)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                            collate_fn=collate_to_max_length)

    model = InteractModel(config.bert_model_path,
                          config.bert_hidden_size,
                          intent_tagset_size,
                          slot_none_tag_size,
                          slot_tag_size,
                          device).to(device)
    # model = torch.nn.DataParallel(model, device_ids=[0, 1], output_device=[0])
    I_criterion = torch.nn.CrossEntropyLoss()
    N_criterion = torch.nn.BCEWithLogitsLoss()

    crf_params = list(map(id, model.CRF.parameters()))  # 把CRF层的参数映射为id
    other_params = filter(lambda x: id(x) not in crf_params, model.parameters())  # 在整个模型的参数中将CRF层的参数过滤掉（filter）

    optimizer = torch.optim.AdamW([{'params': model.CRF.parameters(), 'lr': config.crf_lr},
                                  {'params': other_params, 'lr': config.lr}], weight_decay=config.weight_decay)
    # optimizer = RAdam(model.parameters(), lr=config.lr, weight_decay=0.0)
    total_step = len(train_loader) // config.batch_size * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_step)

    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot.
    epochs_count = []
    train_losses = []
    valid_losses = []

    # Continuing training from a checkpoint if one was given as argument.
    if config.checkpoint:
        checkpoint = torch.load(config.checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]

        print("\t* Training will continue on existing model from epoch {}..."
              .format(start_epoch))

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]

    # Compute loss and accuracy before starting (or resuming) training.
    valid_time, valid_loss, intent_accuracy, slot_none, slot, sen_acc = valid(model,
                                                                              dev_loader,
                                                                              I_criterion,
                                                                              N_criterion)
    print("-> Valid time: {:.4f}s loss = {:.4f} intentAcc: {:.4f} slot_none: {:.4f} slot_F1: {:.4f} SEN_ACC: {:.4f}"
          .format(valid_time, valid_loss, intent_accuracy, slot_none[2], slot[2], sen_acc))

    # -------------------- Training epochs ------------------- #
    print("\n",
          20 * "=",
          "Training Model model on device: {}".format(device),
          20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, config.epochs+1):
        epochs_count.append(epoch)
        print("* Training epoch {}:".format(epoch))
        train_time, train_loss = train(model,
                                       train_loader,
                                       optimizer,
                                       I_criterion,
                                       N_criterion,
                                       config.max_grad_norm)
        train_losses.append(train_loss)
        print("-> Training time: {:.4f}s loss = {:.4f}"
              .format(train_time, train_loss))
        with open('../../user_data/output_model/InteractModel_1/metric.txt', 'a', encoding='utf-8') as fp:
            fp.write('Epoch:' + str(epoch) + '\t' + 'Loss:' + str(round(train_loss, 4)) + '\t')

        valid_time, valid_loss, intent_accuracy, slot_none, slot, sen_acc = valid(model,
                                                                                  dev_loader,
                                                                                  I_criterion,
                                                                                  N_criterion,
                                                                                  )
        print("-> Valid time: {:.4f}s loss = {:.4f} intentAcc: {:.4f} slot_none: {:.4f} slot_F1: {:.4f} SEN_ACC: {:.4f}"
              .format(valid_time, valid_loss, intent_accuracy, slot_none[2], slot[2], sen_acc))
        with open('../../user_data/output_model/InteractModel_1/metric.txt', 'a', encoding='utf-8') as fp:
            fp.write('Loss:' + str(round(valid_loss, 4)) + '\t' + 'Intent_acc:' +
                     str(round(intent_accuracy, 4)) + '\t' + 'slot_none:' +
                     str(round(slot_none[2], 4)) + '\t''slot_F1:' + str(round(slot[2], 4)) + '\t'
                     + 'SEN_ACC:' + str(round(sen_acc, 4)) + '\n')

        valid_losses.append(valid_losses)
        # Update the optimizer's learning rate with the scheduler.
        scheduler.step()

        # Early stopping on validation accuracy.
        if valid_loss > best_score:
            patience_counter += 1
        else:
            best_score = valid_loss
            patience_counter = 0
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                       os.path.join(config.target_dir, "model_best.pth.tar"))

        # Save the model at each epoch.
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "best_score": best_score,
                    "optimizer": optimizer.state_dict(),
                    "epochs_count": epochs_count,
                    "train_losses": train_losses,
                    "valid_losses": valid_losses},
                   os.path.join(config.target_dir, "model_{}.pth.tar".format(epoch)))

        if patience_counter >= config.patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break


if __name__ == "__main__":
    main()
