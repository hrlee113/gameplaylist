import os
import numpy as np
import pandas as pd

'''
Load
'''

def userloader(filename='steam_user_meta_data_final_v1.csv'):
    user = pd.read_csv(os.path.join('data', filename))
    return user




