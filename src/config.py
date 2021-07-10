'''
@Descripttion: 
@version: 
@Author: Yinan Mei
@Date: 2020-07-18 18:31:16
LastEditors: Yinan Mei
LastEditTime: 2021-07-10 17:11:29
'''
import argparse
import json
import os
import logging
import transformers
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, XLNetTokenizer, \
    XLNetForSequenceClassification, XLNetConfig, XLMForSequenceClassification, XLMConfig, XLMTokenizer, \
    RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig, RobertaModel, DistilBertConfig, \
    DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertModel

from logging_customized import setup_logging

setup_logging()

print(transformers.__version__)

class Config():
    DATA_PREFIX = "data"
    EXPERIMENT_PREFIX = "experiments"

    MODEL_CLASSES = {
        'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
        'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
        'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
        'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
        'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    }


def write_config_to_file(args, model_output_dir: str, experiment_name: str):
    config_path = os.path.join(model_output_dir, experiment_name, "args.json")

    with open(config_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def read_arguments():
    parser = argparse.ArgumentParser(description='Run training with following arguments')

    parser.add_argument('--data_dir', default=None, type=str, required=True)
    parser.add_argument('--model_name_or_path', default="pre_trained_model/bert-base-uncased", type=str, required=True)
    parser.add_argument('--model_type', default='roberta', type=str)
    parser.add_argument('--do_lower_case', action='store_true', default=True)
    parser.add_argument('--max_seq_length', default=128, type=int)
    parser.add_argument('--train_batch_size', default=8, type=int)
    parser.add_argument('--eval_batch_size', default=8, type=int)
    parser.add_argument('--num_epochs', default=3.0, type=float)
    parser.add_argument('--save_model_after_epoch', action='store_true')
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--adam_eps', default=1e-8, type=float)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--missing_rate', default=10, type=int)
    parser.add_argument('--neg_num', default=3, type=int)
    parser.add_argument('--select_method', default="top1", type=str)
    parser.add_argument('--nn_num', default=1000, type=int)
    parser.add_argument('--select_threshold', default=0.0, type=float)
    parser.add_argument('--training_ratio', default=1.0, type=float)
    parser.add_argument('--pretrained', default="pre", type=str)
    parser.add_argument('--file_ix', default="all", type=str)
    parser.add_argument('--oov_method', default="bpe", type=str)

    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    args.data_path = os.path.join(Config.DATA_PREFIX, args.data_dir)
    args.model_output_dir = Config.EXPERIMENT_PREFIX

    logging.info("*** parsed configuration from command line and combine with constants ***")

    for argument in vars(args):
        logging.info("argument: {}={}".format(argument, getattr(args, argument)))

    return args