'''
Descripttion: 
version: 
Author: Yinan Mei
Date: 2020-09-02 13:58:28
LastEditors: Yinan Mei
LastEditTime: 2021-07-10 17:40:58
'''
import csv
import logging
import os
from typing import Callable

import pandas as pd

import constants as C
import random
from logging_customized import setup_logging
from copy import deepcopy
from tqdm import tqdm, trange
import argparse
from model import load_model
from config import read_arguments, write_config_to_file, Config

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
        '''
        raise NotImplementedError()

    def get_impute_data(self):
        raise NotImplementedError()

class CategoryDataGenerator(DataGenerator):
    def __init__(self, data_path, k=1000, threshold=50):
        self.data = pd.read_csv(data_path, dtype='str')

    def get_truth_data(self, cat_attr):
        truth_data = self.data[self.data[cat_attr].notna()]
        cols = [col for col in self.data.columns if col not in ['id', cat_attr]]
        truth_data['text_a'] = truth_data[cols].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)
        truth_data = truth_data[['id', 'text_a', cat_attr]]
        truth_data = truth_data.rename(columns={cat_attr:'label'})
        truth_data['label'] = truth_data['label'].apply(lambda x:self.value_to_ix[cat_attr][x])
        return truth_data

    def get_mis_data(self, cat_attr):
        mis_data = self.data[self.data[cat_attr].isna()]
        cols = [col for col in self.data.columns if col not in ['id', cat_attr]]
        mis_data['text_a'] = mis_data[cols].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)
        mis_data = mis_data[['id', 'text_a']]
        return mis_data
    
    def get_domain(self, attr):
        domain = set(self.data[attr])
        if np.nan in domain:
            domain.remove(np.nan)
        value_to_ix, ix_to_value = dict(), dict()
        for ix, v in enumerate(domain):
            value_to_ix[v] = str(ix)
            ix_to_value[ix] = v
        return value_to_ix, ix_to_value
        
    def generate_candidate_df(self, row, candidates):
        tuples = []
        for cand in candidates:
            t = deepcopy(row)
            t.text_b = cand
            tuples.append(t)
        return pd.DataFrame(tuples)

    def get_train_data(self, ignore_attr=['id'], neg_num=5, training_ratio=1.0):
        category_attributes = ATTR_DICT[args.dataset]
        attr_train_data = dict()
        self.value_to_ix, self.ix_to_value = dict(), dict()
        for cat_attr in category_attributes:
            logging.info(f'Generate Training Data for Attribute: {cat_attr}')

            attr_value_to_ix, attr_ix_to_value = self.get_domain(cat_attr)
            self.value_to_ix[cat_attr], self.ix_to_value[cat_attr] = attr_value_to_ix, attr_ix_to_value
            
            truth_data = self.get_truth_data(cat_attr=cat_attr)
            # Sample training data
            truth_data = truth_data.sample(n=int(training_ratio * len(truth_data)), random_state=1234)
            train_df, valid_df = truth_data[:int(len(truth_data)*0.8)], truth_data[int(len(truth_data)*0.8):]
            
            attr_train_data[cat_attr] =(train_df.reset_index().drop('index',axis=1),\
                valid_df.reset_index().drop('index',axis=1))
        return attr_train_data

    def get_impute_data(self, ignore_attr=['id']):
        category_attributes = ATTR_DICT[args.dataset]
        attr_impute_data = dict()
        for cat_attr in category_attributes:
            mis_data = self.get_mis_data(cat_attr=cat_attr)
            attr_impute_data[cat_attr] = mis_data
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

def train(model, train_examples, eval_examples, tokenizer, device, label_list, args=None):
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

def impute_category(data, model, tokenizer, device, args):
    model.eval()
    vocab = tokenizer.get_vocab()
    processor = BERTProcessor()
    examples = processor.get_test_examples(data, with_text_b=False, oov_method=args.oov_method, vocab=vocab)
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
    impute_ixs = torch.argmax(logits, dim=1).cpu().numpy()
    return impute_ixs
    
def category_imputation(data_path, device, args):
    '''
    @Description: Impute the missing attribute one by one.
    @param {type} 
    @return: 
    '''
    cat_data_generator = CategoryDataGenerator(data_path=data_path, k=args.nn_num)
    attr_train_data = cat_data_generator.get_train_data(ignore_attr=['id'], neg_num=args.neg_num, training_ratio=args.training_ratio)
    attr_impute_data = cat_data_generator.get_impute_data(ignore_attr=['id'])
    imputed_table = deepcopy(cat_data_generator.data)
    training_time = dict()
    imputation_time = dict()
    for attr in attr_train_data:
        logging.info(f'Train the model for Attribute: {attr}')
        ix_to_value = cat_data_generator.ix_to_value[attr]
        label_list = [str(ix) for ix in range(len(ix_to_value))]
        config_class, model_class, tokenizer_class = Config.MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(args.model_name_or_path)
        config.num_labels = len(label_list)
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        vocab = tokenizer.get_vocab()
        # model = model_class.from_pretrained(args.model_name_or_path, config=config)
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
        processor = BERTProcessor()
        train_examples = processor.get_train_examples(train_df, with_text_b=False, oov_method=args.oov_method, vocab=vocab)
        eval_examples = processor.get_dev_examples(valid_df, with_text_b=False, oov_method=args.oov_method, vocab=vocab)
        train_start_time = time.process_time()
        best_model = train(model, train_examples, eval_examples, tokenizer, device, label_list=label_list, args=args)
        train_end_time = time.process_time()
        training_time[attr] = train_end_time-train_start_time

        to_impute_data = attr_impute_data[attr]
        imputation_start_time = time.process_time()
        impute_ixs = impute_category(to_impute_data, best_model, tokenizer, device, args)
        ix = 0
        for _, row in tqdm(to_impute_data.iterrows(),total=len(to_impute_data), desc=f"Impute Attribute - {attr}"):
            value = ix_to_value[impute_ixs[ix]]
            if len(value) == 0:
                continue
            row_index = imputed_table[imputed_table.id == row.id].index[0]
            imputed_table.loc[row_index, attr] = value
            ix += 1
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
            
        imputed_table.to_csv(os.path.join(args.data_dir, f'multi_filled_{args.missing_rate}_{cnt}.csv'), index=False)
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

        res_df = pd.DataFrame(res_dic)
        res_df['total'] = res_df.sum(axis=1)
        
        res_df.loc['Impute Accuracy', 'total'] = res_df.loc['Correct Num', 'total'] / res_df.loc['Missing Num', 'total']
        res_df.loc['Impute Ratio', 'total'] = res_df.loc['Impute Num', 'total'] / res_df.loc['Missing Num', 'total']
        res_df.loc['Impute Accuracy (not-na)', 'total'] = (res_df.loc['Correct Num', 'total'] / res_df.loc['Impute Num', 'total']) if res_df.loc['Impute Num', 'total'] > 0 else 0

        if not os.path.exists(f'./output/{args.dataset}/'):
            os.makedirs(f'./output/{args.dataset}/')
        
        print(res_df)

        res_df.to_csv(f'./output/{args.dataset}/res_{args.missing_rate}_{cnt}_multi.csv')

    

    