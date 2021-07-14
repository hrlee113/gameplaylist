import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import save_pickle, torch_device, CustomDataset

# AutoEncoder
class AE(nn.Module):
    def __init__(self):
      super(AE, self).__init__()

      self.encoder = nn.Sequential(
          nn.Linear(426, 512),
          nn.ReLU(),
          nn.Linear(512, 256),
          nn.ReLU(),
          nn.Linear(256, 128)
      )

      self.decoder = nn.Sequential(
          nn.Linear(128, 256),
          nn.ReLU(),
          nn.Linear(256, 512),
          nn.ReLU(),
          nn.Linear(512, 426)
      )

    def forward(self, x):
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)
      return encoded, decoded


def train(model, train_loader, optimizer, criterion, DEVICE):
    model.train()
    train_loss = 0

    for batch_idx, (feature) in enumerate(train_loader):

      feature = feature.to(DEVICE)
      target = feature.to(DEVICE)
      optimizer.zero_grad()
      encoded, decoded = model(feature)
      loss = criterion(decoded, target)
      loss.backward()
      optimizer.step()
      
      train_loss += loss.item()

    train_loss /= len(train_loader)
    return train_loss


def evaluate(model, train_loader, DEVICE):
    model.eval()
    encoded_result = []
    decoded_result = []

    with torch.no_grad():

      for feature in train_loader:
        feature = feature.to(DEVICE)
        encoded, decoded = model(feature)

        encoded_result.append(encoded.cpu().numpy())
        decoded_result.append(decoded.cpu().numpy())

    encoded_result = np.concatenate(encoded_result)
    decoded_result = np.concatenate(decoded_result)
    return encoded_result, decoded_result


def ae_run(game):
    game['clean_genres'] = game['clean_genres'].apply(lambda x: eval(x))
    te = TransactionEncoder()
    te_ary = te.fit(game['clean_genres'].values).transform(game['clean_genres'].values)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    df.iloc[:, :] = df.values.astype(int)
    genre_metrix = df.values

    # hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 20
    DEVICE = torch_device()

    model = AE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_dataset = CustomDataset(genre_metrix)
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, drop_last = False)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, criterion, DEVICE)
        print(f"\n[EPOCH: {epoch}], \tTrain Loss: {train_loss:.4f}")

    final_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = False, drop_last = False)
    AE_encoded_result, AE_decoded_result = evaluate(model, final_loader, DEVICE)

    # 결과(장르 벡터) 저장
    save_pickle(AE_encoded_result, 'data/genres_vecs.pickle')
