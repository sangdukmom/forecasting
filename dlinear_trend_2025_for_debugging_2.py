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

from model.net import DLinear_with_emb as Model
from dataset import Multi_Dataset, Multi_Dataset_2

from config import *

target_col = 'Y6'
date = '250227'
task = 'trend'
model_name = 'dlinear'

def trainer(n_epochs, model_cfg, train_x, train_y, device,
            lr=1e-3, weight_decay=1e-6, grad_max=5):
    loss_fn = torch.nn.MSELoss(reduction='none')
    model = Model(**model_cfg).to(device)
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

if target_col =='Y6':
    meta_df = pd.read_csv("C:/Users/SKSiltron/Desktop/project/signpost_project/1_data/wafer/data_trend/20250210_trend.csv")

meta_df.set_index('date',inplace=True)
meta_df.index = pd.to_datetime(meta_df.index)
meta_df = meta_df.interpolate(method='linear', limit_area='inside') 
meta_df = meta_df.fillna(method='bfill')
meta_df = meta_df.fillna(method='ffill')

base_col = list(meta_df.columns)
base_col.remove("Y6")

dir_out = f'{date}_{target_col}_{task}_auto_regressive'
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

use_col = base_col + [target_col] #expand_col

start_date = "2019-01"
train_end_date = "2024-12"
forecast_start_date = "2025-01"
forecast_end_date = "2025-12"

df = meta_df.loc[start_date:train_end_date, use_col]
df_train = df.copy()

y_scaler = StandardScaler() #StandardScaler 
x_scaler = StandardScaler() #MinMaxScaler

# TODO 1. data input 을 위해 scaler input 수정
x_scaler.fit(df)
y_scaler.fit(df)


model_cfg = cfg['model_cfg']
model_cfg['input_size'] = len(use_col)

# TODO 2. data input 을 위해 scaler input 수정 (2)
data_x = x_scaler.transform(df)
data_y = y_scaler.transform(df)

batch_size = len(data_x) - model_cfg['seq_length'] + 1

dataset = Multi_Dataset_2(x_data=data_x, y_data=data_y, seq_len=model_cfg['seq_length'])
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
    losses, model = trainer(n_epochs, model_cfg, seq_x, seq_y, device)

    # ------------------------------------------------
    # 1.  Teacher Forcing 방식 : 학습데이터 구간 예측 (2019-01 ~ 2024-12)
    # ------------------------------------------------

    # data_x 와 data_y 는 이미 scaling된 전체 학습 데이터 (df 구간)
    teacher_dataset = Multi_Dataset_2(x_data=data_x, y_data=data_y, seq_len=model_cfg['seq_length'])
    # __len__ 이 (len(data_y)-seq_length+1) 로 수정되었으므로, 전체 teacher forcing 예측 window를 한 번에 처리 (?) <-- Multidataset_2 에서 해당부분 수정하지 않음, 코드 디버깅을 위한 메모
    teacher_loader = DataLoader(teacher_dataset, batch_size=len(data_x) - model_cfg['seq_length'] + 1, shuffle=False)
    teacher_batch = next(iter(teacher_loader))['x'] # shape: (num_windows, seq_length, 23)
    teacher_batch = teacher_batch.float().to(device)
    model.eval()
    with torch.no_grad():
        teacher_preds = model(teacher_batch) # shape (num_windows, 1, 23)
    teacher_preds = teacher_preds.squeeze(1).cpu().numpy() # shape (num_windows, 23)

    # teacher_preds[i] corresponds to the prediction for the time at index (i + seq_length - 1)
    # (즉, 첫 window [0:seq_length]의 예측은 시각상 seq_length 번째 시점)
    teacher_dates = meta_df.loc[start_date:train_end_date, use_col].index[model_cfg['seq_length']:]
    teacher_preds_inv = y_scaler.inverse_transform(teacher_preds)

    # ------------------------------------------------
    # 2. Auto-regressive 방식 : Forecast 구간 예측 (2025-01 ~ 2025-12)
    # ------------------------------------------------

    # training data 의 마지막 seq_length 시퀀스를 시작으로 사용
    current_seq = torch.tensor(data_x[-model_cfg['seq_length']:]).unsqueeze(0).float().to(device) # shape (1, seq_length, 23)
    forecast_steps = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='MS')
    auto_preds = [] # auto-regressive 저장 리스트

    with torch.no_grad():
        for step in range(len(forecast_steps)):
            # 현 시퀀스를 입력해 다음 1스텝 예측 (출력 shape: (1, 1, 23))
            pred = model(current_seq)
            # (1, 1, 23) -> (23,) 로 변환
            pred_val = pred.squeeze(0).squeeze(0)
            auto_preds.append(pred_val.cpu().numpy())
            # 새 시퀀스 업데이트 : 가장 오래된 시점 시퀀스 제거, 새로운 예측값 추가
            current_seq = torch.cat([current_seq[:, 1:, :], pred.unsqueeze(1)], dim=1)
    
    auto_preds = np.array(auto_preds) # shape: (forecast_steps, 23)
    forecast_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='MS')
    auto_preds_inv = y_scaler.inverse_transform(auto_preds)

    # ------------------------------------------------
    # 3. (옵션) 타겟 컬럼(Y6)만 추출하여 시각화용 데이터 구성하기
    # ------------------------------------------------

    # 현재 모든 컬럼을 예측하지만, 나중에 Y6 컬럼에 대해 시각화할 예정이므로 해당 컬럼 인덱스를 추출
    target_index = use_col.index(target_col)
    teacher_target_pred = teacher_preds_inv[:, target_index]
    auto_target_pred = auto_preds_inv[:, target_index]

    # 이제 teacher_target_pred 는 2019-01 ~ 2024-12 기간 (예측 window 에 따른 시점)
    # auto_target_pred 는 2025-01 ~ 2025-12 기간에 대한 auto-regressive 예측값
    # 필요에 따라, 두 구간의 예측값을 DataFrame 등으로 결합해서 시각화 가능

    #  result of forecast
    # start_date = "2012-01"
    # end_date = "2024-06"

    # ------------------------------------------------
    # 4. 시각화
    # ------------------------------------------------
    
    # (1) DataFrame 생성 : 학습 구간(teacher forcing) 예측 결과 DataFrame
    df_teacher = pd.DataFrame(teacher_preds_inv, index=teacher_dates, columns=use_col)

    # (2) DataFrame 생성 : 학습 구간 정답(ground truth) DataFrame
    # teacher_dates에 해당하는 meta_df의 데이터를 가져오기
    df_actual = meta_df.loc[teacher_dates, use_col]

    # (3) 테스트 구간(auto-regressive) 예측 결과 DataFrame
    df_forecast = pd.DataFrame(auto_preds_inv, index=forecast_dates, columns=use_col)

    # (4) 시각화를 위한 DataFrame 결합
    # 예를 들어, 대상 컬럼(target_col, 예: 'Y6')의 경우,
    # 학습 구간은 정답과 teacher forcing 예측값, 테스트 구간은 auto-regressive 예측값을 사용

    # 학습 구간 시각화 데이터 : teacher forcing 예측값과 실제값
    df_train_vis = pd.DataFrame({
        'Actual': df_actual[target_col],
        'Predicted': df_teacher[target_col]
    })

     # 테스트 구간 시각화 데이터: auto-regressive 예측값 (정답은 없으므로 예측만)
    df_test_vis = df_forecast[[target_col]].rename(columns={target_col:'Predicted'})
    # 전체 시각화를 위해 학습 구간과 테스트 구간을 날짜 기준으로 결합
    df_vis = pd.concat([df_train_vis, df_test_vis])

    # (5) 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(df_train_vis.index, df_train_vis['Actual'], label='Actual (Train)', marker='o')
    plt.plot(df_train_vis.index, df_train_vis['Predicted'], label='Predicted (Teacher Forcing)', marker='x')
    plt.plot(df_test_vis.index, df_test_vis['Predicted'], label='Predicted (Auto-regressive)', marker='s')
    plt.xlabel('Date')
    plt.ylabel(target_col)
    plt.title(f"{target_col} - Actual vs Predicted")
    plt.legend()
    plt.grid(True)
    
    # 이미지 저장
    plt.savefig(f'{dir_out}/{date}_{target_col}_{task}_{model_name}_{iid}.png')

    plt.show()

    # 가중치, df 저장
    
    torch.save(model.state_dict(), f'{dir_out}/{date}_{target_col}_{task}_{model_name}_{iid}.pth')
    df_vis.to_csv(f'{dir_out}/{date}_{target_col}_{task}_{model_name}_{iid}.csv', index=False, encoding='cp949')

    # ------------------------------------------------
    # 5. 민감도 분석
    # ------------------------------------------------

    # 기존 iter 루프 내에서 예측 코드 이후, gradient 민감도 분석 수행
    # teacher forcing 입력창 중 대표로 마지막 window를 사용

    # teacher forcing 방식의 입력창 생성:
    # 위에서 생성한 teacher_dataset, teacher_loader, teacher_batch 활용
    
    # teacher_dataset = Multi_Dataset_2(x_data=data_x, y_data=data_y, seq_len=model_cfg['seq_length'])
    # teacher_loader = DataLoader(teacher_dataset, batch_size=len(data_x) - model_cfg['seq_length'] + 1, shuffle=False)
    # teacher_batch = next(iter(teacher_loader))['x']
    # teacher_batch = teacher_batch.float().to(device)

    # 대표 입력창 선택 : 마지막 window (또는 원하는 window 선택)
    input_window = teacher_batch[-1]
