from utils import load_pickle
from prep.game_prep import gameloader
# from prep.review_prep import allreviewloader
from prep.dataloader import dataloader
from nlp.lda import genre_lda, content_lda, review_lda
from nlp.sentiment import sentiment_analysis
# from embedding.game2vec import game2vec
from model.gmf import gmf_run
from model.ncf import ncf_run
from model.nmf import nmf_run
from model.dcn import dcn_p_run, dcn_s_run
from model.deepfm import deepfm_run


if __name__ == '__main__':
    # 1. Data load
    # ----- review (게임, 유저, 리뷰가 병합된 데이터)
    train_modified, val_modified, test_modified = dataloader() # split version
    # all_review = allreviewloader()
    # ----- game (LDA를 위한 게임 메타정보 데이터)
    game = gameloader()


    # 2. NLP
    # ----- LDA
    content_topic = content_lda(game); genre_topic = genre_lda(game)
    train_modified = review_lda(train_modified, content_topic, genre_topic)
    val_modified = review_lda(val_modified, content_topic, genre_topic)
    test_modified = review_lda(test_modified, content_topic, genre_topic)
    # ----- Sentiment Analysis
    train_modified = sentiment_analysis(train_modified)
    val_modified = sentiment_analysis(val_modified)
    test_modified = sentiment_analysis(test_modified)


    # 3. Game Embedding
    # _ = game2vec(all_review, game)
    gamevec = load_pickle('norm_game2vec.pickle')


    # 4. model

    # ----- gmf
    _, gmf_acc, gmf_auc, gmf_f1 = gmf_run(train_modified, val_modified, test_modified, gamevec)
    print('========== GMF Score ==========')
    print('ACC : {:.4f}'.format(gmf_acc))
    print('AUC : {:.4f}'.format(gmf_auc))
    print('F1 Score : {:.4f}'.format(gmf_f1))

    # ----- ncf
    _, ncf_acc, ncf_auc, ncf_f1 = ncf_run(train_modified, val_modified, test_modified, gamevec)
    print('========== NCF Score ==========')
    print('ACC : {:.4f}'.format(ncf_acc))
    print('AUC : {:.4f}'.format(ncf_auc))
    print('F1 Score : {:.4f}'.format(ncf_f1))

    # ----- nmf
    nmf_acc, nmf_auc, nmf_f1 = nmf_run(train_modified, val_modified, test_modified, gamevec)
    print('========== NMF Score ==========')
    print('ACC : {:.4f}'.format(nmf_acc))
    print('AUC : {:.4f}'.format(nmf_auc))
    print('F1 Score : {:.4f}'.format(nmf_f1))

    # ----- deepfm
    deepfm_acc, deepfm_auc, deepfm_f1 = deepfm_run(train_modified, val_modified, test_modified, gamevec)
    print('========== DeepFM Score ==========')
    print('ACC : {:.4f}'.format(deepfm_acc))
    print('AUC : {:.4f}'.format(deepfm_auc))
    print('F1 Score : {:.4f}'.format(deepfm_f1))

    # ----- dcn (parallel)
    dcn_p_acc, dcn_p_auc, dcn_p_f1 = dcn_p_run(train_modified, val_modified, test_modified, gamevec)
    print('========== DCN_parallel Score ==========')
    print('ACC : {:.4f}'.format(dcn_p_acc))
    print('AUC : {:.4f}'.format(dcn_p_auc))
    print('F1 Score : {:.4f}'.format(dcn_p_f1))

    # ----- dcn (stacked)
    dcn_s_acc, dcn_s_auc, dcn_s_f1 = dcn_s_run(train_modified, val_modified, test_modified, gamevec)
    print('========== DCN_stacked Score ==========')
    print('ACC : {:.4f}'.format(dcn_s_acc))
    print('AUC : {:.4f}'.format(dcn_s_auc))
    print('F1 Score : {:.4f}'.format(dcn_s_f1))