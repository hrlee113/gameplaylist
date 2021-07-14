import os
import re
import numpy as np
import pandas as pd

'''
Preprocessing
'''

def clear_genres(text): # 장르 전처리 (for LDA)
    text = re.sub('\[', '', text)
    text = re.sub('\]', '', text)
    text = re.sub("\'", '', text)
    return text 

def clear_title(text): # 타이틀 전처리 (for LDA)
    res = re.compile(re.escape("\"")+'.*')
    text = res.sub('', text)
    res = re.compile(re.escape(":")+'.*')
    text = res.sub('', text)
    res = re.compile(re.escape("-")+'.*')
    text = res.sub('', text)
    return text

def remove_title(data): # title 제거한 content (for LDA)
    clean_title = data['title'].apply(clear_title)
    clean_content = data['content'].apply(lambda x : x.lower())
    title_list = []
    for title in clean_title:
        title_list.append(title.lower())
    res = []
    for idx in range(len(clean_content)):
        title = clean_title[idx]
        content = clean_content[idx]
        text = re.sub('[^a-zA-Z0-9]', ' ', content)
        text = re.sub('  ', ' ', text)
        res.append(text)
    return res

def mean_owners(text): # 게임 메타정보 변수생성
    res = re.sub(',', '', text)
    res = re.sub('\.', '',  res).split('  ')
    res = [int(i) for i in res]
    res = np.mean(res)
    return res

'''
Load
'''

def gameloader(filename='steam_game_meta_data_final.csv'):
    data = pd.read_csv(os.path.join('data', filename))
    game = data.drop(['name', 'genres', 'appid', 'score_rank'], axis=1)
    game = game.drop_duplicates().reset_index(drop=True)
    game['mean_owners'] = game['owners'].apply(mean_owners)
    publisher_100 = list(game.publisher.value_counts().index[:100])
    game['publisher_100'] = game['publisher'].apply(lambda x : x if x in publisher_100 else 'etc')
    developer_100 = list(game.developer.value_counts().index[:100])
    game['developer_100'] = game['developer'].apply(lambda x : x if x in developer_100 else 'etc')
    return game




