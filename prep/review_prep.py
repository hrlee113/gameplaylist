import os
import re
import pandas as pd

'''
Load
'''

def reviewloader(filename='steam_reviews_clean_result'):
    train = pd.read_csv(os.path.join('data', '{}_train_v2.csv'.format(filename)), low_memory=False)
    val = pd.read_csv(os.path.join('data', '{}_val_v2.csv'.format(filename)))
    test = pd.read_csv(os.path.join('data', '{}_test_v2.csv'.format(filename)))
    return train, val, test

def allreviewloader(filename='steam_reviews_clean_result_v2.csv'):
    data = pd.read_csv(os.path.join('data', filename))
    return data
