import numpy as np
import torch
import torch.nn as nn
from model.model_prep import run

'''
Model
'''

class GMF_and_NCF(nn.Module):
    def __init__(self, user_num, factor_num):
        super(GMF_and_NCF, self).__init__()

        # GMF 임베딩
        self.GMF_user_embedding = nn.Embedding(user_num, factor_num)

        # NCF 임베딩
        self.NCF_user_embedding = nn.Embedding(user_num, factor_num)

        # NCF_FC_layer
        self.NCF_FC_layer = nn.Sequential(
            nn.Linear(factor_num * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # FC_layer
        self.FC_layer = nn.Sequential(
            nn.Linear(factor_num + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self._init_weight_()

    def _init_weight_(self):
        # weight 초기화
        nn.init.normal_(self.GMF_user_embedding.weight, std=0.01)
        nn.init.normal_(self.NCF_user_embedding.weight, std=0.01)

        for m in self.NCF_FC_layer:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

        for m in self.FC_layer:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, user_idx, item_embedding):
        # GMF
        GMF_user_embedding = self.GMF_user_embedding(user_idx)
        GMF_output = (GMF_user_embedding * item_embedding)

        # NCF
        NCF_user_embedding = self.NCF_user_embedding(user_idx)
        NCF_output = torch.cat((NCF_user_embedding, item_embedding), -1)
        NCF_output = self.NCF_FC_layer(NCF_output)

        output = torch.cat((GMF_output, NCF_output), -1)
        output = self.FC_layer(output)

        return output.view(-1)

'''
Run
'''
    
def nmf_run(train_modified, val_modified, test_modified, gamevec):
    test_loss, test_accuracy, test_auc, test_f1 = run('NMF', train_modified, val_modified, test_modified, gamevec)
    return test_loss, test_accuracy, test_auc, test_f1




