import numpy as np
import torch
import torch.nn as nn
from model.model_prep import run



class GMF(nn.Module):
    def __init__(self, user_num, factor_num):
        super(GMF, self).__init__()

        # 유저 임베딩
        self.user_embedding = nn.Embedding(user_num, factor_num)

        # FC
        self.FC_layer = nn.Sequential(
            nn.Linear(factor_num, 1),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            # nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self._init_weight_()

    def _init_weight_(self):
        # weight 초기화
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        for m in self.FC_layer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, user_idx, item_embedding):
        user_embedding = self.user_embedding(user_idx)
        element_wise_product = (user_embedding * item_embedding)
        out = self.FC_layer(element_wise_product)
        return out.view(-1)


def gmf_run(train_modified, val_modified, test_modified, gamevec):
    test_loss, test_accuracy, test_auc, test_f1 = run('GMF', train_modified, val_modified, test_modified, gamevec)
    return test_loss, test_accuracy, test_auc, test_f1