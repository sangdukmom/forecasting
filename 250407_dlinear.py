import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from model.net import DLinear_with_emb as Model
from dataset import Multi_Dataset
from config import cfg
from tqdm import tqdm

# 설정
target_col = 'Y6'
trend_csv_path = '250407_sarima_selected_trend.csv'
smoothing_value = 4
dir_out = f'250407_Y6_trend_pred_2025_span_{smoothing_value}'
os.makedirs(dir_out, exist_ok=True)

# Y6 trend에서 사용하는 X 인자 prefix 목록
y6_trend_x_prefixes = [
    "D99_WW_KU", "F7_WW_KU", "F10_WW_KU", "X5", "X21", "X48_2", "X49_1",
    "X49_19", "X49_994", "X49_995", "X50_998", "X78_NEW", "X84", "X92",
    "X177_1", "X188_5", "X190_2", "X195_1", "PC_trend", "SMP_trend",
    "BEV_trend", "SERVER_trend"
]

# 데이터 로딩
meta_df = pd.read_csv(trend_csv_path)
meta_df.set_index('Date', inplace=True)
meta_df.index = pd.to_datetime(meta_df.index)
meta_df = meta_df.interpolate(method='linear', limit_area='inside').bfill().ffill()

# prefix 기반 X 컬럼 추출
existing_cols = meta_df.columns.tolist()
final_x_columns = []
for x in y6_trend_x_prefixes:
    matched = [col for col in existing_cols if col.startswith(x + "_")]
    if matched:
        final_x_columns.append(matched[0])
    else:
        print(f"[경고] {x} 컬럼 없음 → 제외됨")

# 입력 컬럼 구성
use_col = final_x_columns + [target_col]

# 학습/예측 구간
start_date = "2019-01"
train_end_date = "2024-12"
forecast_start_date = "2025-01"
forecast_end_date = "2025-12"

# 학습 데이터 구성
df = meta_df.loc[start_date:train_end_date, use_col].copy()
x_scaler = StandardScaler()
y_scaler = StandardScaler()
x_scaler.fit(df.drop(columns=[target_col]))
y_scaler.fit(df[[target_col]])
data_x = x_scaler.transform(df.drop(columns=[target_col]))
data_y = y_scaler.transform(df[[target_col]])

# 슬라이딩 윈도우 구성
model_cfg = cfg['model_cfg']
model_cfg['input_size'] = len(use_col) - 1
seq_len = model_cfg['seq_length']
batch_size = len(data_x) - seq_len + 1
dataset = Multi_Dataset(x_data=data_x, y_data=data_y, seq_len=seq_len)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
seq_x = next(iter(data_loader))['x']
seq_y = next(iter(data_loader))['y']

# 디바이스 및 반복 설정
n_epochs = 500
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
iters = 100
forecast_len = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='MS').shape[0]

# 학습 함수
def trainer(n_epochs, model_cfg, train_x, train_y, device, lr=1e-3, weight_decay=1e-6, grad_max=5):
    model = Model(**model_cfg).to(device)
    loss_fn = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    model.train()
    for epoch in range(n_epochs):
        outputs = model(train_x.float().to(device))
        loss = loss_fn(outputs, train_y.float().to(device)).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_max)
        optimizer.step()
        scheduler.step(loss)
    return loss.item(), model

# 반복 학습
for iid in range(iters):
    print(f"[{iid+1}/100] 학습 중...")
    loss, model = trainer(n_epochs, model_cfg, seq_x, seq_y, device)

    # 예측용 입력 구성
    forecast_df = meta_df.loc[forecast_start_date:forecast_end_date, use_col].copy()
    forecast_df[target_col] = np.nan
    input_df = pd.concat([df, forecast_df])
    forecast_x = x_scaler.transform(input_df.drop(columns=[target_col]))
    forecast_dataset = Multi_Dataset(forecast_x, np.zeros((len(forecast_x), 1)), seq_len=seq_len)
    forecast_loader = DataLoader(forecast_dataset, batch_size=len(forecast_x) - seq_len + 1, shuffle=False)
    forecast_seq_x = next(iter(forecast_loader))['x']

    # 예측 수행
    model.eval()
    with torch.no_grad():
        forecast_pred = model(forecast_seq_x.float().to(device))
    forecast_pred_inv = y_scaler.inverse_transform(forecast_pred.detach().cpu())

    # 시각화 저장
    plt.figure(figsize=(12, 5))
    plt.plot(input_df.index[:len(df)], input_df[target_col].values[:len(df)], label='real')
    plt.plot(input_df.index[-len(forecast_pred_inv):], forecast_pred_inv, label='forecast')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{dir_out}/trend_dlinear_{iid}.png')
    plt.close()

    # 결과 저장
    torch.save(model.state_dict(), f'{dir_out}/trend_dlinear_model_{iid}.pth')
    df_out = pd.DataFrame({
        'date': input_df.index[-len(forecast_pred_inv):].strftime('%Y-%m'),
        'pred': forecast_pred_inv.flatten()
    })
    df_out.to_csv(f'{dir_out}/trend_dlinear_{iid}.csv', index=False, encoding='cp949')
