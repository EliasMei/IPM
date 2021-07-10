'''
@Descripttion: 
@version: 
@Author: Yinan Mei
@Date: 2020-07-17 16:50:26
LastEditors: Yinan Mei
LastEditTime: 2021-07-10 14:04:34
'''
from enum import Enum

import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler

from logging_customized import setup_logging
from feature_extraction import convert_examples_to_features
import logging

setup_logging()


class DataType(Enum):
    TRAINING = "Training"
    EVALUATION = "Evaluation"
    TEST = "Test"

class CustomDataset(TensorDataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)

class CustomLoader(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for idx in range(len(self.dataset)):
            batch = self.dataset[idx]
            yield batch


def load_data(examples, label_list, tokenizer, max_seq_length, batch_size, data_type: DataType, model_type, output_mode="classification"):
    # logging.info("***** Convert Data to Features (Word-Piece Tokenizing) [{}] *****".format(data_type))
    features = convert_examples_to_features(examples,
                                            label_list,
                                            max_seq_length,
                                            tokenizer,
                                            output_mode=output_mode,
                                            cls_token_at_end=bool(model_type in ['xlnet']),
                                            # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=2 if model_type in ['xlnet'] else 0,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(model_type in ['roberta']),
                                            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                            pad_on_left=bool(model_type in ['xlnet']),  # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if model_type in ['xlnet'] else 0,
                                            )

    # logging.info("***** Build PyTorch DataLoader with extracted features [{}] *****".format(data_type))
    # logging.info("Num examples = %d", len(examples))
    # logging.info("Batch size = %d", batch_size)
    # logging.info("Max Sequence Length = %d", max_seq_length)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "without_label":
        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids) 
    else:
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    if data_type == DataType.TRAINING:
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)

    return DataLoader(data, sampler=sampler, batch_size=batch_size)
