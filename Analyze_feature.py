import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.net import DLinear_with_emb as Model
from config import cfg

# ========== 사용자 설정 ==========
trend_model_path = "경로/seasonal_Dlinear_xx.pth"
seasonal_model_path = "경로/trend_Dlinear_xx.pth"
seasonal_x_data_path = "경로/select_feature_seasonal.csv"
trend_x_data_path = "경로/select_feature_trend.csv"
y6_result_path = "경로/250415_Dlinear_y6.csv"
output_dir = "경로/output"
top_n = 20
os.makedirs(output_dir, exist_ok=True)

# ========== 1. 데이터 로드 ==========
seasonal_x_data = pd.read_csv(seasonal_x_data_path)
trend_x_data = pd.read_csv(trend_x_data_path)
y6_result_df = pd.read_csv(y6_result_path)

# 날짜 처리
for df in [seasonal_x_data, trend_x_data, y6_result_df]:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

# 2025-12까지 자르기
seasonal_x_data = seasonal_x_data.loc[:'2025-12']
trend_x_data = trend_x_data.loc[:'2025-12']
y6_result_df = y6_result_df.loc[:'2025-12']

# X 컬럼 목록 추출
seasonal_x_names = [col for col in seasonal_x_data.columns if col not in ['Unnamed: 0', 'Y6_1_1_2']]
trend_x_names = [col for col in trend_x_data.columns if col not in ['Unnamed: 0', 'Y6_2_0_2']]

# ========== 2. 모델 로드 및 가중치 추출 ==========
def load_fc1_fc2_weights(path, input_size):
    model_cfg = cfg['model_cfg'].copy()
    model_cfg['input_size'] = input_size
    model = Model(**model_cfg)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    w_fc1 = model.fc1.weight.detach().cpu().numpy().flatten()
    w_fc2 = model.fc2.weight.detach().cpu().numpy().flatten()
    return w_fc1 + w_fc2

w_trend = load_fc1_fc2_weights(trend_model_path, len(trend_x_names))
w_seasonal = load_fc1_fc2_weights(seasonal_model_path, len(seasonal_x_names))

df_trend = pd.DataFrame({'X': trend_x_names, 'trend_weight': w_trend})
df_seasonal = pd.DataFrame({'X': seasonal_x_names, 'seasonal_weight': w_seasonal})
df_merged = pd.merge(df_trend, df_seasonal, on='X', how='outer').fillna(0)
df_merged['total_weight'] = df_merged['trend_weight'] + df_merged['seasonal_weight']
df_merged['abs_total_weight'] = np.abs(df_merged['total_weight'])
df_sorted = df_merged.sort_values(by='abs_total_weight', ascending=False)
df_sorted.to_csv(f"{output_dir}/dlinear_total_weight_summary.csv", index=False, encoding='cp949')

# ========== 3. 시계열 기여도 계산 ==========
def compute_variable_contributions(X_df, weight_dict):
    contrib_df = pd.DataFrame(index=X_df.index)
    for col in X_df.columns:
        weight = weight_dict.get(col, 0)
        contrib_df[col] = X_df[col] * weight
    contrib_df['y_component_estimated'] = contrib_df.sum(axis=1)
    return contrib_df

# 가중치 딕셔너리
trend_weight_dict = pd.Series(df_sorted['trend_weight'].values, index=df_sorted['X']).to_dict()
seasonal_weight_dict = pd.Series(df_sorted['seasonal_weight'].values, index=df_sorted['X']).to_dict()

# 시계열 기여도
trend_contrib = compute_variable_contributions(trend_x_data[trend_x_names], trend_weight_dict)
seasonal_contrib = compute_variable_contributions(seasonal_x_data[seasonal_x_names], seasonal_weight_dict)

# 최종 예측 재구성
y_total_est = trend_contrib['y_component_estimated'] + seasonal_contrib['y_component_estimated']
combined_contrib = trend_contrib.drop(columns='y_component_estimated') + seasonal_contrib.drop(columns='y_component_estimated')
combined_contrib['y_total_estimated'] = y_total_est
combined_contrib['y6_actual_pred'] = y6_result_df['y6']  # 실제 dlinear 예측값

# 저장
combined_contrib.to_csv(f"{output_dir}/dlinear_contributions_timeseries.csv", encoding='cp949")

# ========== 4. 시각화 ==========
def plot_contributions_lines(contrib_df, top_n=10, title='Top Variable Contributions (Line)', output_path=None):
    top_vars = contrib_df.drop(columns=['y_total_estimated', 'y6_actual_pred']).abs().mean().sort_values(ascending=False).head(top_n).index.tolist()
    colors = plt.cm.tab20(np.linspace(0, 1, top_n))
    plt.figure(figsize=(14, 6))
    for i, var in enumerate(top_vars):
        plt.plot(contrib_df.index, contrib_df[var], label=var, color=colors[i])
    plt.plot(contrib_df.index, contrib_df['y_total_estimated'], label='Estimated Y', color='black', linestyle='--', linewidth=2)
    plt.plot(contrib_df.index, contrib_df['y6_actual_pred'], label='DLinear Y6', color='red', linestyle='-', linewidth=1.5)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()

def plot_contributions_subplots(contrib_df, top_n=6):
    top_vars = contrib_df.drop(columns=['y_total_estimated', 'y6_actual_pred']).abs().mean().sort_values(ascending=False).head(top_n).index.tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, top_n))
    fig, axes = plt.subplots(nrows=top_n, ncols=1, figsize=(12, 2.5 * top_n), sharex=True)
    for i, var in enumerate(top_vars):
        axes[i].plot(contrib_df.index, contrib_df[var], label=var, color=colors[i])
        axes[i].legend()
        axes[i].grid(True)
    plt.tight_layout()
    plt.show()

# 실행
plot_contributions_lines(combined_contrib, top_n=8, output_path=f"{output_dir}/top8_contribution_line.png")
plot_contributions_subplots(combined_contrib, top_n=6)
