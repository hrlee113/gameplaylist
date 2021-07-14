import numpy as np
import torch
import torch.nn as nn
from model.model_prep import run



class NCF(nn.Module):
    def __init__(self, user_num, factor_num):
        super(NCF, self).__init__()

        # 유저 임베딩
        self.user_embedding = nn.Embedding(user_num, factor_num)

        # FC
        self.FC_layer = nn.Sequential(
            nn.Linear(factor_num * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
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
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, user_idx, item_embedding):
        user_embedding = self.user_embedding(user_idx)
        concat_two_latent_vactors = torch.cat((user_embedding, item_embedding), -1)
        out = self.FC_layer(concat_two_latent_vactors)
        return out.view(-1)


def ncf_run(train_modified, val_modified, test_modified, gamevec):
    test_loss, test_accuracy, test_auc, test_f1 = run('NCF', train_modified, val_modified, test_modified, gamevec)
    return test_loss, test_accuracy, test_auc, test_f1