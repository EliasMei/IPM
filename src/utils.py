'''
Descripttion: 
version: 
Author: Yinan Mei
Date: 2021-07-10 13:59:04
LastEditors: Yinan Mei
LastEditTime: 2021-07-10 16:18:08
'''

import pandas as pd

def full_join_except_id(row: pd.Series):
    single_text_blob = []
    for index, row_value in row[1:].items():
        if not row.isnull()[index]:
            single_text_blob.append(str(row_value))

    return " ".join(single_text_blob)