import numpy as np
from embedding.image_cae import cae_run
from embedding.text_ae import ae_run
from embedding.sequence_prod2vec import prod2vec_run
from utils import load_pickle, save_pickle

'''
게임 이미지 & 장르 & sequence 임베딩 생성
'''

def game2vec(all_review, game):
    # ----- image vector
    _ = cae_run(game)
    # ----- genre(text) vector
    _ = ae_run(game)
    # ----- sequence vector
    _ = prod2vec_run(all_review)
    
    img_vecs = load_pickle('data/img_vecs.pickle')
    gen_vecs = load_pickle('data/genres_vecs.pickle')
    seq_vecs = load_pickle('data/seq_vecs.pickle')

    norm_img_vecs = seq_vecs / np.linalg.norm(img_vecs, axis = 1).reshape(-1,1)
    norm_gen_vecs = seq_vecs / np.linalg.norm(gen_vecs, axis = 1).reshape(-1,1)
    norm_seq_vecs = seq_vecs / np.linalg.norm(seq_vecs, axis = 1).reshape(-1,1)
    norm_game2vec = np.concatenate([norm_seq_vecs, norm_img_vecs, norm_gen_vecs], axis = 1)

    _ = save_pickle(norm_game2vec, 'data/norm_game2vec.pickle')
    
    
    
    
    
