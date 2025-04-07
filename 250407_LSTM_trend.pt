import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

from model.net import DLinear_with_emb  # 모델 폴더에 있어도 충돌 없음
from dataset import Multi_Dataset

# LSTM 모델 정의
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, dropout_rate):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)).to(x.device)
        c_0 = Variable(torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)).to(x.device)
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        out = self.fc(hn[-1])
        return out

# 설정값
cfg = {
    'model_cfg': {
        'dropout_rate': 0.2,
        'num_classes': 1,
        'seq_length': 12,
        'hidden_size': 50,
        'num_layers': 1,
        'input_size': 0  # 뒤에서 자동 설정
    },
    'lr': 4e-3,
    'decay': 1e-6,
    'n_epochs': 100,
    'use_cuda': True,
    'cuda_device_num': 0
}

# 파일 경로 수정하세요
trend_csv_path = 'YOUR_PATH/250407_sarima_selected_trend.csv'
target_prefix = 'Y6'
x_prefixes = [  # Y6 트렌드용 X 리스트
    "D99_WW_KU", "F7_WW_KU", "F10_WW_KU", "X5", "X21", "X48_2", "X49_1",
    "X49_19", "X49_994", "X49_995", "X50_998", "X78_NEW", "X84", "X92",
    "X177_1", "X188_5", "X190_2", "X195_1", "PC_trend", "SMP_trend",
    "BEV_trend", "SERVER_trend"
]

# 데이터 로딩
df = pd.read_csv(trend_csv_path, encoding='cp949', sep='\t')
df.columns = [col.strip() for col in df.columns]
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.interpolate(method='linear', limit_area='inside').bfill().ffill()

# 타겟 & 입력 컬럼 자동 추출
target_candidates = [col for col in df.columns if col == target_prefix or col.startswith(target_prefix + "_")]
target_col = target_candidates[0]
x_cols = []
for x in x_prefixes:
    match = [col for col in df.columns if col.startswith(x + "_") or col == x]
    if match:
        x_cols.append(match[0])

cfg['model_cfg']['input_size'] = len(x_cols)

# 날짜 설정
start_date = "2019-01"
train_end_date = "2024-12"
forecast_start_date = "2025-01"
forecast_end_date = "2025-12"

# 학습 데이터 준비
use_col = x_cols + [target_col]
df_train = df.loc[start_date:train_end_date, use_col]
x_scaler = StandardScaler()
y_scaler = StandardScaler()
x_scaler.fit(df_train[x_cols])
y_scaler.fit(df_train[[target_col]])

x_data = x_scaler.transform(df_train[x_cols])
y_data = y_scaler.transform(df_train[[target_col]])

dataset = Multi_Dataset(x_data, y_data, seq_len=cfg['model_cfg']['seq_length'])
data_loader = DataLoader(dataset, batch_size=len(x_data) - cfg['model_cfg']['seq_length'] + 1, shuffle=False)
seq_x = next(iter(data_loader))['x']
seq_y = next(iter(data_loader))['y']

device = torch.device(f'cuda:{cfg["cuda_device_num"]}' if cfg['use_cuda'] and torch.cuda.is_available() else 'cpu')

# 학습 함수
def trainer(n_epochs, model_cfg, train_x, train_y, device, lr, weight_decay, grad_max=5):
    model = LSTM(**model_cfg).to(device)
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
    return model

# 저장 경로
smoothing_value = 4
dir_out = f'LSTM_Y6_trend_pred_2025_span_{smoothing_value}'
os.makedirs(dir_out, exist_ok=True)

# 반복 학습 & 예측
iters = 100
for iid in range(iters):
    print(f"[{iid+1}/100] LSTM 학습 중...")
    model = trainer(cfg['n_epochs'], cfg['model_cfg'], seq_x, seq_y, device, cfg['lr'], cfg['decay'])

    # 예측용 데이터 구성
    df_future = df.loc[forecast_start_date:forecast_end_date, use_col].copy()
    df_future[target_col] = np.nan
    df_input = pd.concat([df_train, df_future])
    forecast_x = x_scaler.transform(df_input[x_cols])
    forecast_dataset = Multi_Dataset(forecast_x, np.zeros((len(forecast_x), 1)), seq_len=cfg['model_cfg']['seq_length'])
    forecast_loader = DataLoader(forecast_dataset, batch_size=len(forecast_x) - cfg['model_cfg']['seq_length'] + 1, shuffle=False)
    forecast_seq_x = next(iter(forecast_loader))['x']

    # 예측 수행
    model.eval()
    with torch.no_grad():
        pred = model(forecast_seq_x.float().to(device))
    pred_inv = y_scaler.inverse_transform(pred.detach().cpu())

    # 시각화
    plt.figure(figsize=(12, 5))
    plt.plot(df_input.index[:len(df_train)], df_input[target_col].values[:len(df_train)], label='real')
    plt.plot(df_input.index[-len(pred_inv):], pred_inv, label='forecast')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{dir_out}/trend_LSTM_{iid}.png')
    plt.close()

    # 저장
    torch.save(model.state_dict(), f'{dir_out}/trend_LSTM_model_{iid}.pth')
    pd.DataFrame({
        'date': df_input.index[-len(pred_inv):].strftime('%Y-%m'),
        'pred': pred_inv.flatten()
    }).to_csv(f'{dir_out}/trend_LSTM_{iid}.csv', index=False, encoding='cp949')
