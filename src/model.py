'''
@Descripttion: 
@version: 
@Author: Yinan Mei
@Date: 2020-07-17 15:42:27
@LastEditors: Yinan Mei
@LastEditTime: 2020-07-28 17:06:42
'''
import os

from config import Config

def save_model(model, experiment_name, model_output_dir, epoch=None, tokenizer=None):
    if epoch:
        output_sub_dir = os.path.join(model_output_dir, experiment_name, "epoch_{}".format(epoch))
    else:
        output_sub_dir = os.path.join(model_output_dir, experiment_name)

    os.makedirs(output_sub_dir, exist_ok=True)

    # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    model_to_save = model.backbone if hasattr(model, 'backbone') else model  # Only save the model it-self
    model_to_save.save_pretrained(output_sub_dir)

    if tokenizer:
        tokenizer.save_pretrained(output_sub_dir)

    return output_sub_dir


def load_model(model_dir, model_type, do_lower_case):
    config_class, model_class, tokenizer_class = Config.MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(model_dir)
    model = model_class.from_pretrained(model_dir)
    tokenizer = tokenizer_class.from_pretrained(model_dir, do_lower_case=do_lower_case)

    return model, tokenizer, config
