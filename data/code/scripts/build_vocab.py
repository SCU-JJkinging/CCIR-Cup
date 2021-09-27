#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/5 15:21
# @Author  : JJkinging
# @File    : build_vocab.py
import json
import pickle

import numpy as np
from collections import Counter

import torch


def build_worddict(train_path, dev_path, test_path):
    '''
    function:构建词典
    :param data: read_data返回的数据
    :return:
    '''
    with open(train_path, 'r', encoding='utf-8') as fp:
        train_data = fp.readlines()
    train_data = [item.strip('\n').split(' ') for item in train_data]
    with open(dev_path, 'r', encoding='utf-8') as fp:
        dev_data = fp.readlines()
    dev_data = [item.strip('\n').split(' ') for item in dev_data]
    with open(test_path, 'r', encoding='utf-8') as fp:
        test_data = fp.readlines()
    test_data = [item.strip('\n').split(' ') for item in test_data]

    word_dict = {}
    words = []
    lengths = []
    for single_data in train_data:
        words.extend(single_data)
        lengths.append(len(single_data))
    for single_data in dev_data:
        words.extend(single_data)
        lengths.append(len(single_data))
    for single_data in test_data:
        words.extend(single_data)
        lengths.append(len(single_data))
    print(len(words))
    print(lengths)
    x = 0
    for len1 in lengths:
        if len1 >= 100:
            x += 1
    print('最长：', max(lengths))
    print('最短：', min(lengths))
    print('长度超过100(含)的句子:', x)

    counts = Counter(words)
    print(counts)
    num_words = len(counts)
    word_dict['[PAD]'] = 0
    word_dict['[CLS]'] = 1
    word_dict['[SEP]'] = 2

    offset = 3

    for i, word in enumerate(counts.most_common(num_words)):
        word_dict[word[0]] = i + offset
    # print(word_dict)
    return word_dict


def build_embedding_matrix(embeddings_file, worddict):
    embeddings = {}
    with open(embeddings_file, "r", encoding="utf8") as input_data:
        for line in input_data:
            line = line.split()
            try:
                float(line[1])
                word = line[0]
                if word in worddict:
                    embeddings[word] = line[1:]

            # Ignore lines corresponding to multiple words separated
            # by spaces.
            except ValueError:
                continue

    num_words = len(worddict)
    embedding_dim = len(list(embeddings.values())[0])
    embedding_matrix = np.zeros((num_words, embedding_dim))

    # Actual building of the embedding matrix.
    missed = 0
    for word, i in worddict.items():
        if word in embeddings:
            embedding_matrix[i] = np.array(embeddings[word], dtype=float)
        else:
            if word == "[PAD]":
                continue
            missed += 1
            # Out of vocabulary words are initialised with random gaussian
            # samples.
            embedding_matrix[i] = np.random.normal(size=(embedding_dim))
    print("Missed words: ", missed)

    return embedding_matrix


def word_to_indices(self, sentence):
    indices = []
    for word in sentence:
        if word in self.word_dict:
            indices.append(self.word_dict[word])
        else:
            indices.append(self.word_dict['UNK'])
    return indices


def load_vocab(label_file):
    '''construct word2id or label2id'''
    vocab = {}
    index = 0
    with open(label_file, 'r', encoding='utf-8') as fp:
        while True:
            token = fp.readline()
            if not token:
                break
            token = token.strip()  # 删除空白符
            vocab[token] = index
            index += 1
    return vocab


if __name__ == "__main__":
    train_path = '../dataset/small_sample/data/train_seq_in.txt'
    dev_path = '../dataset/final_data/dev_seq_in.txt'
    test_path = '../script/test/dataset/test_seq_in.txt'
    word_dict = build_worddict(train_path, dev_path, test_path)
    print(word_dict)
    print(len(word_dict))

    # label_file = '../dataset/new_data/tag.txt'
    # label_dict = load_vocab(label_file)
    # print(label_dict)
    # 保存embedding_file
    embedding_file = '../dataset/embeddings/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5'
    embed_matrix = build_embedding_matrix(embedding_file, word_dict)
    with open("../dataset/embeddings/embedding.pkl", "wb") as pkl_file:
        pickle.dump(embed_matrix, pkl_file)
    print(torch.tensor(embed_matrix).shape)