from itertools import repeat
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import roc_auc_compute_fn, f1_score_compute_fn, torch_device
from model.gmf import GMF
from model.ncf import NCF
from model.nmf import GMF_and_NCF
from model.dcn import DCN_PARALLEL, DCN_STACKED

# ============== Collaborative Filtering ==============

class GMFNCFData(Dataset):
    def __init__(self, user_id_idx_li, game_id_idx_li, label_li):
      super(GMFNCFData, self).__init__()
      self._user_id_idx_li = user_id_idx_li
      self._game_id_idx_li = game_id_idx_li
      self._label_li = label_li

    def __len__(self):
      return len(self._label_li)

    def __getitem__(self, idx):
      user = self._user_id_idx_li[idx]
      game = self._game_id_idx_li[idx]
      label = self._label_li[idx]
      return user, game, label


def train(model, train_loader, optimizer, criterion, DEVICE, gamevec):
    model.train()
    train_loss = 0; correct = 0
    output_li = []; label_li = []

    for user, game, label in train_loader:
        user = user.to(DEVICE)
        item_embedding = gamevec[game].to(DEVICE)
        label = label.to(DEVICE)
        label = label.type(torch.DoubleTensor)

    optimizer.zero_grad()
    output = model(user, item_embedding)
    output = output.type(torch.DoubleTensor)

    loss = criterion(output, label)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    output = (output > 0.5).float()
    correct += (output == label).float().sum()

    output_li.append(output.detach().cpu().numpy())
    label_li.append(label.detach().cpu().numpy())

    output_li = np.concatenate(output_li)
    label_li = np.concatenate(label_li)

    train_loss /= len(train_loader)
    train_accuracy = 100. * correct / len(train_loader.dataset)
    train_auc = roc_auc_compute_fn(output_li, label_li)
    train_f1 = f1_score_compute_fn(output_li, label_li)

    return train_loss, train_accuracy, train_auc, train_f1


def evaluate(model, test_loader, criterion, DEVICE, gamevec):
    model.eval()
    test_loss = 0; correct = 0
    output_li = []; label_li = []

    with torch.no_grad():
        for user, game, label in test_loader:

            user = user.to(DEVICE)
            item_embedding = gamevec[game].to(DEVICE)
            label = label.to(DEVICE)
            label = label.type(torch.DoubleTensor)

            output = model(user, item_embedding)
            output = output.type(torch.DoubleTensor)

            loss = criterion(output, label)

            test_loss += loss.item()

            output = (output>0.5).float()
            correct += (output == label).float().sum()

            output_li.append(output.detach().cpu().numpy())
            label_li.append(label.detach().cpu().numpy())

    output_li = np.concatenate(output_li)
    label_li = np.concatenate(label_li)

    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    test_auc = roc_auc_compute_fn(output_li, label_li)
    test_f1 = f1_score_compute_fn(output_li, label_li)

    return test_loss, test_accuracy, test_auc, test_f1


def run(model_text, train_modified, val_modified, test_modified, gamevec, model_save=False):
    train_user_id_idx_li = train_modified['label_encode_user_id'].tolist()
    train_game_id_idx_li = train_modified['label_encode_game_id'].tolist()
    train_label_li = train_modified['label'].astype(float).tolist()

    val_user_id_idx_li = val_modified['label_encode_user_id'].tolist()
    val_game_id_idx_li = val_modified['label_encode_game_id'].tolist()
    val_label_li = val_modified['label'].astype(float).tolist()

    test_user_id_idx_li = test_modified['label_encode_user_id'].tolist()
    test_game_id_idx_li = test_modified['label_encode_game_id'].tolist()
    test_label_li = test_modified['label'].astype(float).tolist()

    torch_norm_game2vec = torch.FloatTensor(gamevec)

    user_num = max(train_modified['label_encode_user_id'].max(), val_modified['label_encode_user_id'].max(), test_modified['label_encode_user_id'].max()) + 1 
    factor_num = gamevec.shape[1]

    # hyperparameters
    BATCH_SIZE = 512
    EPOCHS = 30
    DEVICE = torch_device()

    # 모델 정의
    if model_text == 'GMF':
        model = GMF(user_num = user_num, factor_num = factor_num).to(DEVICE)
    elif model_text == 'NCF':
        model = NCF(user_num = user_num, factor_num = factor_num).to(DEVICE)
    elif model_text == 'NMF':
        model = GMF_and_NCF(user_num = user_num, factor_num = factor_num).to(DEVICE)
    elif model_text == 'DCN_PARALLEL':
        model = DCN_PARALLEL(user_num = user_num, factor_num = factor_num).to(DEVICE)
    elif model_text == 'DCN_STACKED':
        model = DCN_STACKED(user_num = user_num, factor_num = factor_num).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCELoss()

    train_dataset = GMFNCFData(train_user_id_idx_li, train_game_id_idx_li, train_label_li)
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, drop_last = False)

    val_dataset = GMFNCFData(val_user_id_idx_li, val_game_id_idx_li, val_label_li)
    val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False, drop_last = False)

    # 학습

    # best_metric = 0; best_epoch = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy, train_auc, train_f1 = train(model, train_loader, optimizer, criterion, DEVICE, torch_norm_game2vec)
        test_loss, test_accuracy, test_auc, test_f1 = evaluate(model, val_loader, criterion, DEVICE, torch_norm_game2vec)
        print(f"[EPOCH: {epoch}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f} %, Train F!-Score: {train_f1:.4f}, Train AUC: {train_auc:.4f}, \
        Val Loss: {test_loss:.4f}, Val Accuracy: {test_accuracy:.2f} %, Val F!-Score: {test_f1:.4f}, Val AUC: {test_auc:.4f} \n")

        if model_save==True:
            if best_metric < test_auc:

                best_metric = test_auc
                best_epoch = epoch
                torch.save(model.state_dict(), 'model/NCF_Best_model_state_dict.pt')
        else:
            pass

    # 평가
    test_dataset = GMFNCFData(test_user_id_idx_li, test_game_id_idx_li, test_label_li)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, drop_last = False)

    # model.load_state_dict(torch.load('model/NCF_Best_model_state_dict.pt'))
    test_loss, test_accuracy, test_auc, test_f1 = evaluate(model, test_loader, criterion, DEVICE, torch_norm_game2vec)
    
    return test_loss, test_accuracy, test_auc, test_f1


# ============== DeepFM ==============

def get_modified_data(X_train, X_valid, X_test, continuous_fields, categorical_fields):
    # 인코딩 이후의 데이터에 대해 각 칼럼이 본래 인코딩 이전에 몇 번째 field에 속했었는지에 대한 정보
    field_dict = dict()
    field_index = []
    X_train_modified = pd.DataFrame()
    X_valid_modified = pd.DataFrame()
    X_test_modified = pd.DataFrame()

    for index, col in enumerate(X_train.columns):

        if col in continuous_fields:
            field_dict[index] = col
            field_index.append(index)

            scaler = MinMaxScaler()
            X_cont_col = pd.DataFrame(scaler.fit_transform(X_train[[col]]), columns=[col])
            X_cont_col1 = pd.DataFrame(scaler.transform(X_valid[[col]]), columns=[col])
            X_cont_col2 = pd.DataFrame(scaler.transform(X_test[[col]]), columns=[col])
            X_train_modified = pd.concat([X_train_modified, X_cont_col], axis=1)
            X_valid_modified = pd.concat([X_valid_modified, X_cont_col1], axis=1)
            X_test_modified = pd.concat([X_test_modified, X_cont_col2], axis=1)

        if col in categorical_fields:

            X_cate_col = pd.get_dummies(X_train[col], prefix=col, prefix_sep='-')
            field_dict[index] = list(X_cate_col.columns)
            field_index.extend(repeat(index, X_cate_col.shape[1]))
            X_cate_col1 = pd.get_dummies(X_valid[col], prefix=col, prefix_sep='-')
            X_cate_col2 = pd.get_dummies(X_test[col], prefix=col, prefix_sep='-')
            for c in X_cate_col.columns:
                if c not in X_cate_col1.columns:
                    X_cate_col1[c] = 0
                if c not in X_cate_col2.columns:
                    X_cate_col2[c] = 0
            X_train_modified = pd.concat([X_train_modified, X_cate_col], axis=1)
            X_valid_modified = pd.concat([X_valid_modified, X_cate_col1], axis=1)
            X_test_modified = pd.concat([X_test_modified, X_cate_col2], axis=1)

    print('Data Prepared...')
    print('X_train shape: {}'.format(X_train_modified.shape))
    print('X_valid shape: {}'.format(X_valid_modified.shape))
    print('X_test shape: {}'.format(X_test_modified.shape))
    print('# of Feature: {}'.format(len(field_index)))
    print('# of Field: {}'.format(len(field_dict)))

    return field_dict, field_index, X_train_modified, X_valid_modified, X_test_modified


def get_data(train_modified, val_modified, test_modified, gamevec, CONT_FIELDS, CAT_FIELDS, BATCH_SIZE):

    X_train = train_modified.drop('recommended', axis=1); Y_train = train_modified.loc[:, 'recommended']
    X_valid = val_modified.drop('recommended', axis=1); Y_valid = val_modified.loc[:, 'recommended']
    X_test = test_modified.drop('recommended', axis=1); Y_test = test_modified.loc[:, 'recommended']

    train_multi = train_modified[['label_encode_game_id']].merge(gamevec, left_on='label_encode_game_id', right_on='index').drop(['label_encode_game_id', 'index'], axis=1)
    val_multi = val_modified[['label_encode_game_id']].merge(gamevec, left_on='label_encode_game_id', right_on='index').drop(['label_encode_game_id', 'index'], axis=1)
    test_multi = test_modified[['label_encode_game_id']].merge(gamevec, left_on='label_encode_game_id', right_on='index').drop(['label_encode_game_id', 'index'], axis=1)

    field_dict, field_index, X_train, X_valid, X_test = get_modified_data(X_train, X_valid, X_test, CONT_FIELDS, CAT_FIELDS)
    X_train = pd.concat([X_train, train_multi], axis=1).dropna()
    X_valid = pd.concat([X_valid, val_multi], axis=1).dropna()
    X_test = pd.concat([X_test, test_multi], axis=1).dropna()

    train_ds = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_train.values, tf.float32), tf.cast(Y_train, tf.float32))) \
        .shuffle(30000).batch(BATCH_SIZE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_valid.values, tf.float32), tf.cast(Y_valid, tf.float32))) \
        .shuffle(10000).batch(BATCH_SIZE)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_test.values, tf.float32), tf.cast(Y_test, tf.float32))) \
        .shuffle(10000).batch(BATCH_SIZE)

    return train_ds, val_ds, test_ds, field_dict, field_index