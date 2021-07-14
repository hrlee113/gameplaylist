import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import save_pickle, get_labels_images, torch_device, CustomDataset

'''
Convolutional AutoEncoder
'''
class AE(nn.Module):
  def __init__(self):
    super(AE, self).__init__()

    self.encoder = nn.Sequential(
        nn.Conv2d(3,16,3,padding=1),   # batch x 16 x 28 x 28
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16,32,3,padding=1),  # batch x 32 x 28 x 28
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(32,64,3,padding=1),  # batch x 32 x 28 x 28
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2,2),   # batch x 64 x 14 x 14
        nn.Conv2d(64,128,3,padding=1),  # batch x 128 x 14 x 14
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2,2),
        nn.Conv2d(128,256,3,padding=1),  # batch x 256 x 7 x 7
        nn.ReLU()
    )

    self.encoder_fc = nn.Sequential(
        nn.Linear(256 * 7 * 7, 128)
    )

    self.decoder_fc = nn.Sequential(
        nn.Linear(128, 256 * 7 * 7)
    )

    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), # batch x 128 x 14 x 14
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(128, 64, 3, 1, 1), # batch x 64 x 14 x 14
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64, 16, 3, 1, 1), # batch x 16 x 14 x 14
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.ConvTranspose2d(16, 3, 3, 2, 1, 1), # batch x 1 x 28 x 28
        nn.ReLU()
    )

  def forward(self, x):
    encoded = self.encoder(x)
    encoded = self.encoder_fc(encoded.view(-1, 256 * 7 * 7))

    decoded = self.decoder_fc(encoded)
    decoded = self.decoder(decoded.view(-1, 256, 7, 7))
    return encoded, decoded.view(-1, 3 * 28 * 28)


def train(model, train_loader, optimizer, criterion, DEVICE):
    model.train() # 모델을 학습상태로 지정
    train_loss = 0
    for batch_idx, (image, _) in enumerate(train_loader):
        image = image.to(DEVICE)
        target = image.view(-1, 3 * 28 * 28).to(DEVICE)
        optimizer.zero_grad()
        encoded, decoded = model(image)
        loss = criterion(decoded, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_loss /= len(train_loader)
    return train_loss


def evaluate(model, test_loader, optimizer, criterion, DEVICE):
    model.eval() # 모델을 평가상태로 지정
    test_loss = 0
    real_image = []
    gen_image = []

    with torch.no_grad(): # 모델을 평가하는 단계에서 기울기를 통해 파라미터 값이 업데이트 되는 현상을 방지하기 위해서 지정, Gradient의 흐름을 억제
        for image, label in test_loader:
            image = image.to(DEVICE)
            target = image.view(-1, 3 * 28 * 28).to(DEVICE)
            optimizer.zero_grad()
            encoded, decoded = model(image)
            test_loss += criterion(decoded, target).item()
            real_image.append(image.cpu().detach().numpy())
            gen_image.append(decoded.cpu().detach().numpy())

    test_loss /= len(test_loader)
    return test_loss, real_image, gen_image

  
'''
run
'''
def cae_run(game):
    labels, images = get_labels_images(game)
    dataset = CustomDataset(x_data = images, y_data = labels)
    # hyperparameters
    BATCH_SIZE = 128
    EPOCHS = 100
    # 디바이스 설정
    DEVICE = torch_device()
    
    # 학습
    train_loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = False)
    test_loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = False)
    model = AE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, criterion, DEVICE)
        test_loss, real_image, gen_image = evaluate(model, test_loader, optimizer, criterion, DEVICE)
        print(f"\n[EPOCH: {epoch}], \tTrain Loss: {train_loss:.4f}, \tTest Loss: {test_loss:.4f}")

    # 평가
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = False)
    model.eval()
    img_vecs = []
    idx2game_id = []
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(dataloader):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            encoded, decoded = model(image)
            img_vecs.append(encoded.cpu().detach().numpy())
            idx2game_id.append(label.cpu().detach().numpy())
    img_vecs = np.concatenate(img_vecs)
    idx2game_id = np.concatenate(idx2game_id)

    # 결과(이미지 벡터) 저장
    save_pickle(img_vecs, 'data/img_vecs.pickle')
