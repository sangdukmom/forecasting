import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model.net import DLinear_with_emb as Model
from config import cfg

# ====== 사용자 설정 ======
trend_model_path = "경로/250415_y6/trend_Dlinear_12.pth"
seasonal_model_path = "경로/250415_y6/seasonal_Dlinear_2.pth"
seasonal_x_data_path = "경로/select_feature_seasonal.csv"
trend_x_data_path = "경로/select_feature_trend.csv"
output_dir = "경로/250415_y6"
top_n = 20

# ====== 데이터 로딩 ======
seasonal_x_data = pd.read_csv(seasonal_x_data_path)
trend_x_data = pd.read_csv(trend_x_data_path)

seasonal_x_data['Date'] = pd.to_datetime(seasonal_x_data['Date'])
trend_x_data['Date'] = pd.to_datetime(trend_x_data['Date'])
seasonal_x_data.set_index('Date', inplace=True)
trend_x_data.set_index('Date', inplace=True)

seasonal_x_data = seasonal_x_data.loc[:'2025-12']
trend_x_data = trend_x_data.loc[:'2025-12']

seasonal_x_names = list(seasonal_x_data.columns)
trend_x_names = list(trend_x_data.columns)

for remove_col in ['Unnamed: 0', 'Y6_1_1_2']:
    if remove_col in seasonal_x_names:
        seasonal_x_names.remove(remove_col)
for remove_col in ['Unnamed: 0', 'Y6_2_0_2']:
    if remove_col in trend_x_names:
        trend_x_names.remove(remove_col)

# ====== 가중치 로딩 함수 ======
def load_fc1_fc2_weights(path, input_size):
    model_cfg = cfg['model_cfg'].copy()
    model_cfg['input_size'] = input_size
    model = Model(**model_cfg)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    w_fc1 = model.fc1.weight.detach().cpu().numpy().flatten()
    w_fc2 = model.fc2.weight.detach().cpu().numpy().flatten()
    return w_fc1 + w_fc2

# ====== 가중치 추출 ======
w_trend = load_fc1_fc2_weights(trend_model_path, len(trend_x_names))
w_seasonal = load_fc1_fc2_weights(seasonal_model_path, len(seasonal_x_names))

df_trend = pd.DataFrame({'X': trend_x_names, 'trend_weight': w_trend})
df_seasonal = pd.DataFrame({'X': seasonal_x_names, 'seasonal_weight': w_seasonal})
df_merged = pd.merge(df_trend, df_seasonal, on='X', how='outer').fillna(0)
df_merged['total_weight'] = df_merged['trend_weight'] + df_merged['seasonal_weight']
df_merged['abs_total_weight'] = np.abs(df_merged['total_weight'])
df_sorted = df_merged.sort_values(by='abs_total_weight', ascending=False)
df_sorted.to_csv(f"{output_dir}/dlinear_total_weight_summary.csv", index=False, encoding='cp949')

# ====== 기여도 계산 ======
def compute_variable_contributions(X_df, weight_dict):
    contrib_df = pd.DataFrame(index=X_df.index)
    for col in X_df.columns:
        weight = weight_dict.get(col, 0)
        contrib_df[col] = X_df[col] * weight
    contrib_df['y_estimated'] = contrib_df.sum(axis=1)
    return contrib_df

df_weights = pd.read_csv(f"{output_dir}/dlinear_total_weight_summary.csv", encoding='cp949')
trend_weight_dict = pd.Series(df_weights['trend_weight'].values, index=df_weights['X']).to_dict()
seasonal_weight_dict = pd.Series(df_weights['seasonal_weight'].values, index=df_weights['X']).to_dict()

trend_contrib = compute_variable_contributions(trend_x_data[trend_x_names], trend_weight_dict)
seasonal_contrib = compute_variable_contributions(seasonal_x_data[seasonal_x_names], seasonal_weight_dict)

# ====== 수정된 합산 방식 ======
combined_contrib = pd.DataFrame(index=trend_contrib.index)

# 1. 실제 예측값 생성
combined_contrib['y_total_estimated'] = (
    trend_contrib['y_estimated'] + seasonal_contrib['y_estimated']
)

# 2. 변수 기여도 합산 (두쪽에 걸쳐 있는 X 인자들도 포함)
all_vars = list(set(trend_contrib.columns).union(set(seasonal_contrib.columns)))
all_vars = [var for var in all_vars if var != 'y_estimated']

for var in all_vars:
    t = trend_contrib[var] if var in trend_contrib else 0
    s = seasonal_contrib[var] if var in seasonal_contrib else 0
    combined_contrib[var] = t + s

# 컬럼 순서 정리
ordered_cols = [col for col in combined_contrib.columns if col != 'y_total_estimated'] + ['y_total_estimated']
combined_contrib = combined_contrib[ordered_cols]

# ====== 시각화 함수 ======
def plot_contributions_lines(contrib_df, top_n=10, title='Variable Contributions (Line)', output_path=None):
    top_vars = contrib_df.drop(columns='y_total_estimated').abs().mean().sort_values(ascending=False).head(top_n).index.tolist()
    plt.figure(figsize=(14, 6))
    for var in top_vars:
        plt.plot(contrib_df.index, contrib_df[var], label=var)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Contribution")
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()

def plot_contributions_subplots(contrib_df, top_n=6):
    top_vars = contrib_df.drop(columns='y_total_estimated').abs().mean().sort_values(ascending=False).head(top_n).index.tolist()
    fig, axes = plt.subplots(nrows=top_n, ncols=1, figsize=(12, 2.5 * top_n), sharex=True)
    for i, var in enumerate(top_vars):
        axes[i].plot(contrib_df.index, contrib_df[var], label=var)
        axes[i].set_title(f"{var} Contribution")
        axes[i].grid(True)
        axes[i].legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# ====== 실행 ======
plot_contributions_lines(combined_contrib, top_n=8, title='Top 8 Variables - Line Plot', output_path='top8_line_plot.png')
plot_contributions_subplots(combined_contrib, top_n=6)
