'''
@Descripttion: 
@version: 
@Author: Yinan Mei
@Date: 2020-07-20 19:37:55
LastEditors: Yinan Mei
LastEditTime: 2020-08-25 23:19:23
'''
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, RobertaConfig

class TripleClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*3, config.hidden_size)
        dropout_prob = config.hidden_dropout_prob if isinstance(config, RobertaConfig) else config.dropout
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class TripleBERT(BertPreTrainedModel):
    def __init__(self, base_model, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.backbone = base_model
        self.classifier = TripleClassificationHead(config)

        self.init_weights()

    def forward(self, inputs_list, labels):
        seq_outputs_list = []
        for inputs in inputs_list:
            part_outputs = self.backbone(
                **inputs
            ) 
            seq_outputs_list.append(part_outputs[0])
        sequence_output = torch.cat(seq_outputs_list, dim=-1)

        logits = self.classifier(sequence_output)

        outputs = (logits)
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss, logits)

        return outputs  # (loss), logits