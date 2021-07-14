from PIL import Image
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import torch
from torch.utils.data import Dataset

'''
Metric
'''

def roc_auc_compute_fn(y_pred, y_true):
    return roc_auc_score(y_true, y_pred)
def f1_score_compute_fn(y_pred, y_true):
    return f1_score(y_true, y_pred)


'''
Pickle
'''

def load_pickle(filename):
    with open(filename, 'rb') as f:
        res = pickle.load(f)
    return res
def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        
'''
Embedding
'''

def get_labels_images(game):
    # 이미지 데이터 로드
    image_dict = {}
    for i in range(5):
        image = load_pickle('image_data_dict_v2_{}.pickle'.format(i+1))
        image_dict.update(image)
    # 게임 고유번호 - 이미지 매칭
    labels=[]; images=[]
    for i in range(len(game)):
        game_id = game.loc[i, 'game_id']
        image = image_dict[game_id]
        image = Image.fromarray(image)
        image = image.resize((28, 28)).convert('RGB')
        image = np.array(image)
        image = image.reshape(1, 28, 28, 3)
        image = np.swapaxes(image, 1, 3)
        labels.append(game_id)
        images.append(image)
    labels = np.array(labels); images = np.concatenate(images)
    return labels, images

class CustomDataset(Dataset):
  # 데이터 정의
  def __init__(self, x_data, y_data = None):
    self.x_data = x_data
    self.y_data = y_data.reshape(-1,1)
  # 총 데이터 개수
  def __len__(self):
    return len(self.x_data)
  # 인덱싱
  def __getitem__(self, idx):
    if self.y_data is None:
      x = torch.FloatTensor(self.x_data[idx])
      return x
    else:
      x = torch.FloatTensor(self.x_data[idx])
      y = torch.LongTensor(self.y_data[idx])[0]
      return x, y


'''
Modeling
'''

def torch_device():
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    return DEVICE




