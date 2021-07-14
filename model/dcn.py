import numpy as np
import torch
import torch.nn as nn
from model.model_prep import run

'''
Model
'''

class DCN_PARALLEL(nn.Module):
    def __init__(self, user_num, factor_num):
        super(DCN_PARALLEL, self).__init__()

        # 유저 임베딩
        self.user_embedding = nn.Embedding(user_num, factor_num)

        # Cross Network
        self.CN1 = nn.Linear(factor_num, factor_num)
        self.CN2 = nn.Linear(factor_num, factor_num)
        self.CN3 = nn.Linear(factor_num, factor_num)

        # Deep Network
        self.DN = nn.Sequential(
            nn.Linear(factor_num, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # FC
        self.FC_layer = nn.Sequential(
            nn.Linear(factor_num + 32, 1),
            nn.Sigmoid()
        )

        self._init_weight_()

    def _init_weight_(self):
        # weight 초기화
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        for m in self.DN:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, user_idx, item_embedding):
        user_embedding = self.user_embedding(user_idx)
        embedding = (user_embedding * item_embedding)

        # Cross Network
        CN_out = self.CN1(embedding) * embedding
        CN_out = self.CN2(CN_out) * embedding
        CN_out = self.CN3(CN_out)

        # Deep Network
        DN_out = self.DN(embedding)
        out = torch.cat((CN_out, DN_out), -1)
        out = self.FC_layer(out)

        return out.view(-1)


class DCN_STACKED(nn.Module):
  def __init__(self, user_num, factor_num):
      super(DCN_STACKED, self).__init__()

      # 유저 임베딩
      self.user_embedding = nn.Embedding(user_num, factor_num)

      # Cross Network
      self.CN1 = nn.Linear(factor_num, factor_num)
      self.CN2 = nn.Linear(factor_num, factor_num)
      self.CN3 = nn.Sequential(
          nn.Linear(factor_num, factor_num),
          nn.ReLU()
      )

      # Deep Network
      self.DN = nn.Sequential(
          nn.Linear(factor_num, 64),
          nn.ReLU(),
          nn.Linear(64, 32),
          nn.ReLU()
      )

      # FC
      self.FC_layer = nn.Sequential(
          nn.Linear(32, 1),
          nn.Sigmoid()
      )

      self._init_weight_()

  def _init_weight_(self):
      # weight 초기화
      nn.init.normal_(self.user_embedding.weight, std=0.01)
      for m in self.DN:
          if isinstance(m, nn.Linear):
              nn.init.xavier_uniform_(m.weight)

  def forward(self, user_idx, item_embedding):
      user_embedding = self.user_embedding(user_idx)

      embedding = (user_embedding * item_embedding)

      # Cross Network
      CN_out = self.CN1(embedding) * embedding
      CN_out = self.CN2(CN_out) * embedding
      CN_out = self.CN3(CN_out)

      # Deep Network
      DN_out = self.DN(CN_out)
      out = self.FC_layer(DN_out)
      return out.view(-1)

'''
Run
'''

def dcn_p_run(train_modified, val_modified, test_modified, gamevec):
    test_loss, test_accuracy, test_auc, test_f1 = run('DCN_PARALLEL', train_modified, val_modified, test_modified, gamevec)
    return test_loss, test_accuracy, test_auc, test_f1

def dcn_s_run(train_modified, val_modified, test_modified, gamevec):
    test_loss, test_accuracy, test_auc, test_f1 = run('DCN_STACKED', train_modified, val_modified, test_modified, gamevec)
    return test_loss, test_accuracy, test_auc, test_f1





