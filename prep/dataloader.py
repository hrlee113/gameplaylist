import re 
import pandas as pd
from game_prep import gameloader
from user_prep import userloader
from review_prep import reviewloader


def data_preprocessing(game, user, review):

    # 게임메타데이터 사용 변수
    GAME_LIST = ['label_encode_game_id', 'topic', 'positive', 'negative', 'publisher_100', 'developer_100', 'userscore', 
    'average_forever', 'average_2weeks', 'median_forever', 'median_2weeks', 'price', 'game_total_cnt', 'game_rec_cnt', 
    'game_not_rec_cnt', 'recommend_ratio', 'not_recommend_ratio', 'game_total_avg_play_time', 'game_rec_avg_play_time', 
    'game_not_rec_avg_play_time', 'initialprice', 'discount', 'ccu', 'num_genres', 'num_languages', 'mean_owners']

    # 리뷰데이터 사용 변수
    REVIEW_LIST = ['label_encode_game_id', 'label_encode_user_id', 'recommended', 'timestamp', 'play_time_minute', 
    'review_time_minute', 'review_helpful_count', 'sentiment', 'content_length', 'content_lda',	'game_topic']

    if type(review['recommended'][0]) != int:
        review['recommended'] = review['recommended'].apply(lambda x : 1 if x=='Recommended' else 0)
    else:
        pass
    review['timestamp'] = review['timestamp'].apply(lambda x : int(re.sub('-', '', x)))
    data = pd.merge(review[REVIEW_LIST], game[GAME_LIST], on='label_encode_game_id', how='left')
    data = pd.merge(data, user.drop('user_id', axis=1), how='left', on='label_encode_user_id')
    data = data.dropna().reset_index(drop=True)
    return data


def dataloader():
    game = gameloader() # 게임 메타 정보
    user = userloader() # 유저 메타 정보
    train, val, test = reviewloader() # 리뷰 데이터
    # 병합
    train_modified = data_preprocessing(game, user, train)
    val_modified = data_preprocessing(game, user, val)
    test_modified = data_preprocessing(game, user, test)
    
    return train_modified, val_modified, test_modified
    