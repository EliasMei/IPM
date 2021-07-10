'''
Descripttion: 
version: 
Author: Yinan Mei
Date: 2020-09-02 13:58:28
LastEditors: Yinan Mei
LastEditTime: 2021-07-10 17:16:35
'''
import csv
import logging
import os
from typing import Callable

import constants as C
import random
from logging_customized import setup_logging
from copy import deepcopy
from tqdm import tqdm, trange
import argparse
from model import load_model
from config import read_arguments, write_config_to_file, Config
from knn_finder import KnnFinder

import torch
import pandas as pd
import numpy as np
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaForSequenceClassification, RobertaTokenizer
from data_representation import InputExample, BERTProcessor
from feature_extraction import convert_examples_to_features
from data_loader import DataType, load_data
from optimizer import build_optimizer
from torch_initializer import initialize_gpu_seed
from evaluation import Evaluation
import time

setup_logging()

def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

ATTR_DICT = {
    'example':['category', 'brand']
}

class DataGenerator(object):
    def get_train_data(self):
        '''
        Description: Get Train/Valid Data for Model Training
        param {type} 
        return {type} 
        '''
        raise NotImplementedError()

    def get_impute_data(self):
        raise NotImplementedError()

class CategoryDataGenerator(DataGenerator):
    def __init__(self, data_path, k=1000, threshold=50):
        self.data = pd.read_csv(data_path, dtype='str')
        # if the K is small, then we may omit some candidates, which may lead to the unrepeatable problem
        knn_finder = KnnFinder(k=k)
        self.knn_dic, self.sim_dic = knn_finder.find_knn(self.data, threshold=threshold)

    def get_truth_data(self, cat_attr):
        truth_data = self.data[self.data[cat_attr].notna()]
        cols = [col for col in self.data.columns if col not in ['id', cat_attr]]
        truth_data['text_a'] = truth_data[cols].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)
        truth_data = truth_data[['id', 'text_a', cat_attr]]
        truth_data = truth_data.rename(columns={cat_attr:'text_b'})
        truth_data['label'] = "1"
        return truth_data

    def get_mis_data(self, cat_attr):
        mis_data = self.data[self.data[cat_attr].isna()]
        cols = [col for col in self.data.columns if col not in ['id', cat_attr]]
        mis_data['text_a'] = mis_data[cols].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)
        mis_data = mis_data[['id', 'text_a', cat_attr]]
        mis_data = mis_data.rename(columns={cat_attr:'text_b'})
        return mis_data

    def negative_sampling_from_knn(self, truth_data, cat_attr, neg_num=5):
        negative_samples = []
        random.seed(1234)
        # avoid out-of-order keys, which leads to the unrepeatable problem
        domain = sorted(self.data[cat_attr].value_counts().index)
        for _, row in tqdm(truth_data.iterrows(), desc="Negative Sampling from KNN", total=len(truth_data)):
            id_, text_a, text_b = row.id, row.text_a, row.text_b
            nn_dic = self.knn_dic[id_].knn_dic
            sorted_nn = sorted(nn_dic.items(), key=lambda x:(x[1],x[0]), reverse=True)
            neg_values = []
            for nn_id, sim in sorted_nn:
                neg_value = self.data[self.data.id == nn_id][cat_attr].values[0]
                if pd.isnull(neg_value):
                    continue
                if neg_value == text_b:
                    continue
                if neg_value in neg_values:
                    continue
                neg_values.append(neg_value)
                if len(neg_values) == neg_num:
                    break
            if len(neg_values) < neg_num:
                while len(neg_values) < neg_num:
                    neg_value = random.choice(domain)
                    if pd.isnull(neg_value):
                        continue
                    if neg_value == text_b:
                        continue
                    if neg_value in neg_values:
                        continue
                    neg_values.append(neg_value)
            for neg_v in neg_values:
                negative_samples.append({'id':id_, 'text_a':text_a, 'text_b':neg_v})
        neg_df = pd.DataFrame(negative_samples)
        neg_df['label'] = "0"
        return neg_df
        
    def generate_candidate_df(self, row, candidates):
        tuples = []
        for cand in candidates:
            t = deepcopy(row)
            t.text_b = cand
            tuples.append(t)
        return pd.DataFrame(tuples)

    def generate_category_candidates(self, row, cat_attr):
        id_ = row.id
        if id_ not in self.knn_dic:
            return set()
        candidates = ['nan']
        nn_ids = sorted(self.knn_dic[id_].knn_dic.keys())
        for nn_id in nn_ids:
            cand_value = str(self.data[self.data.id == nn_id][cat_attr].values[0])
            if cand_value in candidates:
                continue
            candidates.append(cand_value)
        candidates.remove('nan')
        return candidates   

    def get_train_data(self, ignore_attr=['id'], neg_num=5, training_ratio=1.0):
        category_attributes = ATTR_DICT[args.dataset]
        attr_train_data = dict()
        for cat_attr in category_attributes:
            logging.info(f'Generate Training Data for Attribute: {cat_attr}')
            truth_data = self.get_truth_data(cat_attr=cat_attr)
            # Sample training data
            truth_data = truth_data.sample(n=int(training_ratio * len(truth_data)), random_state=1234)
            train_truth_data, valid_truth_data = truth_data[:int(len(truth_data)*0.8)], truth_data[int(len(truth_data)*0.8):]

            train_neg_df = self.negative_sampling_from_knn(train_truth_data, cat_attr, neg_num=neg_num)
            valid_neg_df = self.negative_sampling_from_knn(valid_truth_data, cat_attr, neg_num=neg_num)

            train_df, valid_df = pd.concat([train_truth_data, train_neg_df]), pd.concat([valid_truth_data, valid_neg_df])
            train_df, valid_df = train_df.sample(n=len(train_df), random_state=1234), valid_df.sample(n=len(valid_df), random_state=1234)
            # train_df.to_csv(f'./train_{cat_attr}', index=False)
            
            attr_train_data[cat_attr] =(train_df.reset_index().drop('index',axis=1),\
                valid_df.reset_index().drop('index',axis=1))
        return attr_train_data

    def get_impute_data(self, ignore_attr=['id']):
        category_attributes = ATTR_DICT[args.dataset]
        attr_impute_data = dict()
        for cat_attr in category_attributes:
            mis_data = self.get_mis_data(cat_attr=cat_attr)
            cand_df_dic = dict()
            cand_num = 0
            for _, row in tqdm(mis_data.iterrows(), desc=f'Generate to-Impute Data for Attribute: {cat_attr}', total=len(mis_data)):
                #* Find candidates from KNN
                candidates = self.generate_category_candidates(row, cat_attr=cat_attr)
                if len(candidates) == 0:
                    continue
                else:
                    cand_num += len(candidates)
                cand_df = self.generate_candidate_df(row, candidates)
                cand_df_dic[row.id] = cand_df
            attr_impute_data[cat_attr] = cand_df_dic
            logging.info(f'Average Num of Candidates: {cand_num/len(mis_data)}')
        return attr_impute_data
        

def train_bert(device,
          train_dataloader,
          model,
          tokenizer,
          optimizer,
          scheduler,
          evaluation,
          num_epochs,
          max_grad_norm,
          model_type):
    logging.info("***** Run training *****")

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    # we are interested in 0 shot learning, therefore we already evaluate before training.
    eval_results = evaluation.evaluate(model, device, -1)

    best_f1 = -1
    best_model = model
    epoch = -1
    for epoch in trange(int(num_epochs), desc="Epoch"):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()

            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}

            if model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            global_step += 1

            logging_loss = tr_loss

        eval_results = evaluation.evaluate(model, device, epoch)
        eval_f1 = eval_results['f1_score']

        if best_f1 < eval_f1:
            best_f1 = eval_f1
            best_model = model
            best_epoch = epoch
            logging.info(f"Best Model Saved. \n Scores on Eval dataset:\n Precision: {eval_results['precision']}; Recall: {eval_results['recall']}; F1-score: {eval_results['f1_score']}")

    return best_model

def train(model, train_df, valid_df, tokenizer, device, args=None):
    vocab = tokenizer.get_vocab()
    processor = BERTProcessor()
    label_list = processor.get_labels()
    train_examples = processor.get_train_examples(train_df, oov_method=args.oov_method, vocab=vocab)
    training_data_loader = load_data(train_examples,
                                     label_list,
                                     tokenizer,
                                     args.max_seq_length,
                                     args.train_batch_size,
                                     DataType.TRAINING, args.model_type)
    logging.info("loaded {} training examples".format(len(train_examples)))

    num_train_steps = len(training_data_loader) * args.num_epochs

    optimizer, scheduler = build_optimizer(model,
                                           num_train_steps,
                                           args.learning_rate,
                                           args.adam_eps,
                                           args.warmup_steps,
                                           args.weight_decay)
    # logging.info("Built optimizer: {}".format(optimizer))

    eval_examples = processor.get_dev_examples(valid_df, oov_method=args.oov_method, vocab=vocab)
    logging.info("loaded and initialized evaluation examples {}".format(len(eval_examples)))
    evaluation_data_loader = load_data(eval_examples,
                                       label_list,
                                       tokenizer,
                                       args.max_seq_length,
                                       args.eval_batch_size,
                                       DataType.EVALUATION, args.model_type)
    evaluation = Evaluation(evaluation_data_loader, None, None, len(label_list), args.model_type)
    best_model = train_bert(device,
          training_data_loader,
          model,
          tokenizer,
          optimizer,
          scheduler,
          evaluation,
          args.num_epochs,
          args.max_grad_norm,
          args.model_type)
    return best_model

def select_candidates(cand_df, model, tokenizer, device, args):
    model.eval()
    cnt = 0
    ix_value_dic = {}
    for _, row in cand_df.iterrows():
        ix_value_dic[cnt] = row.text_b
        cnt += 1
    vocab = tokenizer.get_vocab()
    processor = BERTProcessor()
    examples = processor.get_test_examples(cand_df, oov_method=args.oov_method, vocab=vocab)
    data_loader = load_data(examples,
                            label_list=None,
                            output_mode='without_label',
                            tokenizer=tokenizer,
                            max_seq_length=args.max_seq_length,
                            batch_size=args.eval_batch_size,
                            data_type=DataType.EVALUATION, model_type=args.model_type)
    
    logits = []
    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)
        
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                    'attention_mask': batch[1]}

            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs, labels=None)
            batch_logits = outputs[0].cpu()
            logits.append(batch_logits)
            del batch
            del outputs
            torch.cuda.empty_cache()
    logits = torch.cat(logits)
    logits = logits.softmax(dim=1)
    if args.select_method == 'top1':
        cand_value = ix_value_dic[torch.argmax(logits, dim=0)[1].item()]
    elif args.select_method == 'all':
        cand_ixs = torch.nonzero(logits[:,1] > args.select_threshold)
        cand_values = [ix_value_dic[ix.item()] for ix in cand_ixs]
        cand_value = " ".join(cand_values)
    elif args.select_method == 'mixed':
        if torch.max(logits[:,1]).item() > args.select_threshold:
            cand_value = ix_value_dic[torch.argmax(logits, dim=0)[1].item()]
        else:
            cand_value = ""
    return cand_value
    
def category_imputation(data_path, device, args):
    '''
    @Description: Impute the missing attribute one by one.
    '''
    cat_data_generator = CategoryDataGenerator(data_path=data_path, k=args.nn_num)
    attr_train_data = cat_data_generator.get_train_data(ignore_attr=['id'], neg_num=args.neg_num, training_ratio=args.training_ratio)
    attr_impute_data = cat_data_generator.get_impute_data(ignore_attr=['id'])
    imputed_table = deepcopy(cat_data_generator.data)
    training_time = dict()
    imputation_time = dict()
    for attr in attr_train_data:
        logging.info(f'Train the model for Attribute: {attr}')
        config_class, model_class, tokenizer_class = Config.MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(args.model_name_or_path)
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        if args.pretrained == "pre":
            print('Pretrianed')
            model = model_class.from_pretrained(args.model_name_or_path, config=config)
        elif args.pretrained == "scratch":
            print('Training from scratch')
            model = model_class(config)
        else:
            raise ValueError("Wrong Argument 'pretrained' parsed!")
        model.to(device)
        
        train_df, valid_df = attr_train_data[attr]
        train_start_time = time.process_time()
        best_model = train(model, train_df, valid_df, tokenizer, device, args)
        train_end_time = time.process_time()
        training_time[attr] = train_end_time-train_start_time

        logging.info(f'Impute Attribute: {attr}')
        to_impute_data = attr_impute_data[attr]
        imputation_start_time = time.process_time()

        for id_, cand_df in tqdm(to_impute_data.items(),total=len(to_impute_data), desc=f"Impute Attribute - {attr}"):
            value = select_candidates(cand_df, best_model, tokenizer, device, args)
            if len(value) == 0:
                continue
            row_index = imputed_table[imputed_table.id == id_].index[0]
            imputed_table.loc[row_index, attr] = value
        imputation_end_time = time.process_time()

        imputation_time[attr] = imputation_end_time-imputation_start_time
    return imputed_table, training_time, imputation_time


if __name__ == "__main__":
    args = read_arguments()
    if args.file_ix == 'all':
        file_ixs = range(5)
    else:
        file_ixs = [int(args.file_ix)]

    for cnt in file_ixs:
        device, n_gpu = initialize_gpu_seed(args.seed)
        
        seed_everything(args.seed)
        mis_data_path = os.path.join(args.data_dir, f'mis_data_{args.missing_rate}_{cnt}.csv')
        
        imputed_table, training_time, imputation_time = category_imputation(data_path=mis_data_path, device=device, args=args)
            
        imputed_table.to_csv(os.path.join(args.data_dir, f'binary_filled_{args.missing_rate}_{cnt}.csv'), index=False)
        logging.info("Imputation done.")

        #! Evaluation
        origin_data = pd.read_csv(os.path.join(args.data_dir, 'origin_data.csv'))
        mis_data = pd.read_csv(mis_data_path)

        cat_attributes = ATTR_DICT[args.dataset]

        # cal scores
        mis_num = mis_data.isna().sum()
        impute_num = mis_num - imputed_table.isna().sum()
        correct_num = (imputed_table == origin_data).sum() - (mis_data == origin_data).sum()

        res_dic = {}
        for cat_attr in cat_attributes:
            attr_res_dic = {}
            imp_acc = correct_num[cat_attr] / mis_num[cat_attr]
            imp_ratio = impute_num[cat_attr] / mis_num[cat_attr]
            imp_acc_notna = (correct_num[cat_attr] / impute_num[cat_attr]) if impute_num[cat_attr] > 0 else 0

            attr_res_dic['Missing Num'] = mis_num[cat_attr]
            attr_res_dic['Impute Num'] = impute_num[cat_attr]
            attr_res_dic['Correct Num'] = correct_num[cat_attr]
            attr_res_dic['Impute Accuracy'] = imp_acc
            attr_res_dic['Impute Ratio'] = imp_ratio
            attr_res_dic['Impute Accuracy (not-na)'] = imp_acc_notna
            attr_res_dic['Training Time'] = training_time[cat_attr]
            attr_res_dic['Imputation Time'] = imputation_time[cat_attr]
            attr_res_dic['Total Time'] = training_time[cat_attr] + imputation_time[cat_attr]

            res_dic[cat_attr] = attr_res_dic

        res_df = pd.DataFrame(res_dic, index=['Missing Num', 'Impute Num', 'Correct Num', 'Impute Accuracy', 'Impute Ratio', 'Impute Accuracy (not-na)', 'Training Time', 'Imputation Time', 'Total Time'])
        res_df['total'] = res_df.sum(axis=1)
        
        res_df.loc['Impute Accuracy', 'total'] = res_df.loc['Correct Num', 'total'] / res_df.loc['Missing Num', 'total']
        res_df.loc['Impute Ratio', 'total'] = res_df.loc['Impute Num', 'total'] / res_df.loc['Missing Num', 'total']
        res_df.loc['Impute Accuracy (not-na)', 'total'] = (res_df.loc['Correct Num', 'total'] / res_df.loc['Impute Num', 'total']) if res_df.loc['Impute Num', 'total'] > 0 else 0

        if not os.path.exists(f'./output/{args.dataset}/'):
            os.makedirs(f'./output/{args.dataset}/')
        
        print(res_df)

        res_df.to_csv(f'./output/{args.dataset}/res_{args.missing_rate}_{cnt}_binary.csv')

    

    