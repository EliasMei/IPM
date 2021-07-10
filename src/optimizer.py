'''
@Descripttion: 
@version: 
@Author: Yinan Mei
@Date: 2020-07-17 15:43:25
@LastEditors: Yinan Mei
@LastEditTime: 2020-07-17 18:21:16
'''
from transformers import AdamW, get_linear_schedule_with_warmup


def build_optimizer(model, num_train_steps, learning_rate, adam_eps, warmup_steps, weight_decay):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)

    return optimizer, scheduler

def build_triple_optimizer(model, num_train_steps, learning_rate, adam_eps, warmup_steps, weight_decay):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in filter(lambda x: x[1].requires_grad, model.named_parameters()) if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in filter(lambda x: x[1].requires_grad, model.named_parameters()) if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)

    return optimizer, scheduler
