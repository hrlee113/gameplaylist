import numpy as np
import pandas as pd
import random
from gensim.models import Word2Vec
from utils import save_pickle

'''
게임 sequence 임베딩 생성 : word2vec
'''

def prod2vec_run(all_review):

    # 선호도를 분류한 데이터 전처리
    gp_user_like = all_review.groupby(['recommended', 'user_id'])
    # 선호 게임과 비선호 게임의 시퀀스를 분리
    rec_user_play_game_li = ([gp_user_like.get_group(gp)['game_id'].astype(str).tolist() for gp in gp_user_like.groups])
    for user_play_game_li in rec_user_play_game_li:
        random.Random(22).shuffle(user_play_game_li)

    rec_yes_shuffle_yes_model = Word2Vec(sentences = rec_user_play_game_li, 
                            iter = 10, min_count = 1, size = 300, workers = 4,
                            sg = 1, hs = 0, negative = 5, window = 9999999)
    # 모델 저장
    # rec_yes_shuffle_yes_model.wv.save_word2vec_format('model/rec_yes_shuffle_yes_model_min_count_1')

    game_id_li = all_review['game_id'].astype(str).tolist()
    seq_vecs = []
    for game_id in game_id_li:
        seq_vecs += [rec_yes_shuffle_yes_model[game_id]]
    seq_vecs = np.concatenate([seq_vecs], axis = 1)

    # 결과(장르 벡터) 저장
    save_pickle(seq_vecs, 'data/seq_vecs.pickle')
