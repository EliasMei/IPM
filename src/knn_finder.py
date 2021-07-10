'''
Descripttion: 
version: 
Author: Yinan Mei
Date: 2020-08-27 17:09:27
LastEditors: Yinan Mei
LastEditTime: 2021-07-10 17:37:05
'''
import csv
import logging
import os
from typing import Callable

import pandas as pd
from tqdm import tqdm

import constants as C
from logging_customized import setup_logging
from utils import full_join_except_id
from collections import defaultdict 
import time
from copy import deepcopy
import argparse

setup_logging()
tqdm.pandas()

def term_jaccard_sim(terms1, terms2):
    return len(terms1.intersection(terms2)) / len(terms1.union(terms2))

def custom_tokenizer(x, specials=['-','/',',', '(', ')', '#', '&', '!']):
    for special in specials:
        x=x.replace(special, ' ')
    words = x.strip().split()
    return words

def preprocess(data):
    joined_data = {}
    for ix, row in tqdm(data.iterrows(), total=len(data), desc='Join conc_df'):
        joined_data[row.id] = custom_tokenizer(full_join_except_id(row))
    return joined_data

class KnnMapper(object):
    def __init__(self, k):
        self.k = k
        self.kmax_sim = 0
        self.kmax_id = None
        self.knn_dic = dict()
        
    def update_kmax(self):
        kmax_id = next(iter(self.knn_dic))
        kmax_sim = self.knn_dic[kmax_id]
        for tmp_id, tmp_sim in self.knn_dic.items():
            if tmp_sim < kmax_sim:
                kmax_sim = tmp_sim
                kmax_id = tmp_id
        self.kmax_sim = kmax_sim
        self.kmax_id = kmax_id
        return
    
    def add(self, nn_id, sim):
        if len(self.knn_dic) < self.k:
            self.knn_dic[nn_id] = sim
            self.update_kmax()
        else:
            if sim > self.kmax_sim:
                del self.knn_dic[self.kmax_id]
                self.knn_dic[nn_id] = sim
                self.update_kmax()
        return

class KnnFinder(object):
    def __init__(self, k):
        self.k = k
        super(KnnFinder, self).__init__()

    def build_inverse_index(self, data, threshold=30):
        word_to_id = defaultdict(lambda:set())
        for id_, words in tqdm(data.items()):
            for word in words:
                word_to_id[word].add(id_)
        del_keys = [k for k,v in word_to_id.items() if len(v) > threshold or len(v) == 1]
        for k in del_keys:
            del word_to_id[k]
        return word_to_id
    
    def find_neighbors(self, word_to_id):
        nn_dic = defaultdict(lambda:set())
        for word, ids in tqdm(word_to_id.items()):
            for id_ in ids:
                nn_dic[id_].update(ids.difference({id_}))
        return nn_dic

    def find_knn(self, data, threshold=30):
        joined_data = preprocess(data)
        joined_set_data = {k:set(v) for k,v in joined_data.items()}
        
        word_to_id = self.build_inverse_index(joined_data, threshold=threshold)
        print('Num of Candidates:', sum([len(v) for k,v in word_to_id.items()]))
        
        nn_dic = self.find_neighbors(word_to_id)
        
        sim_dic = defaultdict(lambda:0)
        knn_dic = defaultdict(lambda:KnnMapper(k=self.k))
        for tid, nn_ids in tqdm(nn_dic.items(), desc='KNN Finding'):
            for nn_id in nn_ids:
                # avoid repeated calculations 
                if (tid, nn_id) in sim_dic:
                    sim = sim_dic[(tid, nn_id)]
                else:
                    sim = term_jaccard_sim(joined_set_data[tid], joined_set_data[nn_id])
                    sim_dic[(tid, nn_id)] = sim
                    sim_dic[(nn_id, tid)] = sim
                knn_dic[tid].add(nn_id, sim)
        return knn_dic, sim_dic