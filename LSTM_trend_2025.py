import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import datetime

import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader

# 내 PC\ECM 드라이브 (S:)\공유 폴더\DS2팀\01. 업무\20. TF signpost\season 3\0. previous\3. PREDICTION\Dlinear model
from model.net import DLinear_with_emb as Model
from dataset import Multi_Dataset

from torch.autograd import Variable

# from config import *

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, dropout_rate):
        super().__init__()
        self.num_classes = num_classes # output size
        self.num_layers = num_layers # number of recurrent layers in the lstm
        self.input_size = input_size # input size
        self.hidden_size = hidden_size # neurons in each lstm layer
        self.seq_length = seq_length
        self.dropout_rate = dropout_rate
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        self.fc =  nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)

        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        # out = self.relu(hn)
        out = self.fc(hn)
        return out

  cfg = {'model_cfg': {
                'dropout_rate':0.2,
                'num_classes':1,
                'seq_length':12,
                'hidden_size':50,
                'num_layers':1,
                'input_size':22 # 입력하는 X인자 개수
                },
      'lr':4e-3,
       'decay':1e-6,
       'n_epochs':100,
       
       'use_cuda':False,
       'cuda_device_num':0,
       
       'start_data':'2018-01',
        'end_data':'2024-09',

        # GA 파라미터
        'population_size':20, # 세대 구성원
        'num_generations':20, # 다음 세대 반복 수
        'crossover_rate':0.8, # 아빠, 엄마 유전자 조합 비율
        'mutation_rate':0.1, # 돌연변이
        'elitism':2 # 엘리트 
      }

target_col = 'Y6'

def trainer(n_epochs, model_cfg, train_x, train_y, device,
            lr=1e-3, weight_decay=1e-6, grad_max=5):
    loss_fn = torch.nn.MSELoss(reduction='none')
    model = LSTM(**model_cfg).to(device)
    optimiser = torch.optim.Adam(model.parameters(), 
                                 lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.1, patience=5)
    model.train()
    losses_x, losses_y = [], []
    train_x = train_x.float().to(device)
    train_y = train_y.float().to(device)
    for epoch in range(n_epochs):   
        outputs = model(train_x)
        optimiser.zero_grad()
        y_loss = loss_fn(outputs, train_y)
        loss = y_loss.mean() 
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_max)
        optimiser.step()

        scheduler.step(loss)
        losses_y.append(loss.item())
    return np.mean(losses_y), model

def tester(model, test_x, test_y, device):
    loss_fn = torch.nn.MSELoss(reduction='none') 
    model.eval()
    # losses_y = []
    with torch.no_grad():
        test_x = test_x.float().to(device)
        test_y = test_y.float().to(device)

        outputs = model(test_x)
        y_loss = loss_fn(outputs, test_y)
        loss = y_loss.mean() 
        RMSE_loss = torch.sqrt(loss)

    return RMSE_loss.item(), outputs.detach().cpu()

meta_df = pd.read_csv("C:/Users/SKSiltron/Desktop/project/signpost_project/1_data/wafer/data_trend/20250210_trend.csv")
meta_df.set_index('date',inplace=True)
meta_df.index = pd.to_datetime(meta_df.index)
meta_df = meta_df.interpolate(method='linear', limit_area='inside') 
meta_df = meta_df.fillna(method='bfill')
meta_df = meta_df.fillna(method='ffill')

base_col = list(meta_df.columns)
base_col.remove("Y6")

smoothing_value = 4
dir_out = f'LSTM_0213_{target_col}_trend_pred_2025_span_{smoothing_value}'
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

# pull train
use_col = base_col + [target_col] #expand_col
# use_col = meta_df.columns
start_date = "2019-01"
train_end_date = "2024-12"
forecast_start_date = "2025-01"
forecast_end_date = "2025-12"
df = meta_df.loc[start_date:train_end_date, use_col]
df_train = df.copy()
# df['EV'] = np.log(df['EV']) + 1 # if EV scaler is True 
# scale
y_scaler = StandardScaler() #StandardScaler 
# y_scaler = StandardScaler() #StandardScaler
# x_scaler = MinMaxScaler() #MinMaxScaler
x_scaler = StandardScaler() #MinMaxScaler

x_scaler.fit(df.drop(target_col, axis=1))
y_scaler.fit(df[[target_col]])

# model_cfg = cfg['model_cfg']
# model_cfg['input_size'] = len(use_col) - 1

data_x = x_scaler.transform(df.drop(target_col, axis=1))
data_y = y_scaler.transform(df[[target_col]])

# batch_size = len(data_x) - cfg['seq_length'] + 1
batch_size = len(data_x) - cfg['model_cfg']['seq_length'] + 1

dataset = Multi_Dataset(x_data=data_x, y_data=data_y, seq_len=cfg['model_cfg']['seq_length'])
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

seq_x = next(iter(data_loader))['x']
seq_y = next(iter(data_loader))['y']

test_size = batch_size // 5
n_epochs = 500
use_cuda = True # GPU 사용 유무
device = torch.device('cuda', 0) if use_cuda else torch.device("cpu")

iters = 100
forcasting_len = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='MS').shape[0] # 총 예측 기간
# forcasting_len = 14 # 24-07 ~ 25-12  총 18개월
for iid in range(iters):
    print(iid)
    losses, model = trainer(n_epochs, cfg['model_cfg'], seq_x, seq_y, device)
    # test
    # start_date = "2018-01"
    # end_date = "2024-06"
    
    # 1. Prepare Input for Forecasting:
   #  forecast_df = meta_df.loc["2024-11":"2025-12", use_col].copy() # 미래 예측 기간
    forecast_df = meta_df.loc[forecast_start_date:forecast_end_date, use_col].copy() # 미래 예측 기간
    
    forecast_df.loc[:, target_col] = np.nan # target 값은 예측할거니까 NaN으로 변경
    
    input_df = pd.concat([df, forecast_df]) # 학습 데이터 + 미래 예측 기간 데이터

    forecast_x = x_scaler.transform(input_df.drop(target_col, axis=1)) # scale
    
    
    # 2. Create Dataset for Forecast
    forecast_dataset = Multi_Dataset(x_data = forecast_x, y_data = np.zeros((len(forecast_x),1)), seq_len=cfg['model_cfg']['seq_length']) # y는 임의의 값
    
    # 3. Prepare Forecast Data
    forecast_loader = DataLoader(forecast_dataset, batch_size=len(forecast_x) - cfg['model_cfg']['seq_length'] + 1, shuffle=False) # data load
    forecast_seq_x = next(iter(forecast_loader))['x'] # x만 필요

    # 4. Make Prediction 
    model.eval()
    with torch.no_grad():
       forecast_seq_x = forecast_seq_x.float().to(device) # cpu or gpu
       forecast_pred = model(forecast_seq_x) # pred
    
    forecast_pred_scale_true = y_scaler.inverse_transform(forecast_pred.detach().cpu()) # Inverse transform
    

    #  result of forecast
    # start_date = "2012-01"
    # end_date = "2024-06"
    
    
   #  input_df[target_col] = input_df[target_col].ewm(span=smoothing_value).mean() # smoothing
    
    plt.figure(figsize=(12, 5))
    
   #  plt.plot(input_df.index[:-forcasting_len], input_df[target_col].values[:-forcasting_len], label='real') # real 값
   #  plt.plot(input_df.index[-forcasting_len:], forecast_pred_scale_true[-forcasting_len:], label='forecasting')  # 예측 값
    plt.plot(input_df.index[:len(df)], input_df[target_col].values[:len(df)], label='real') # real 값
    plt.plot(input_df.index[-len(forecast_pred_scale_true):], forecast_pred_scale_true, label='forecasting')  # 예측 값
    # plt.xticks(x, df.index, rotation=90)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{dir_out}/trend_LSTM_{iid}.png')
    plt.show()

    torch.save(model.state_dict(), f'{dir_out}/trend_LSTM_model_0213_{iid}.pth')

    df_save = pd.DataFrame()
   #  df_save['date'] = input_df.index[-forcasting_len:]
    df_save['date'] = input_df.index[-len(forecast_pred_scale_true):]
    df_save['date'] = df_save['date'].dt.strftime('%Y-%m')
    df_save['pred'] = forecast_pred_scale_true.flatten()


    df_save.to_csv(f'{dir_out}/trend_LSTM_0213_{iid}.csv', index=False, encoding='cp949')

