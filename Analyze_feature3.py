import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.net import DLinear_with_emb as Model
from config import cfg
from scipy.signal import find_peaks

# ==============================
# 1. 경로 및 데이터 로드
# ==============================

# --- 사용자 경로 설정 (수정해주세요) ---
trend_model_path = "your_trend_model.pth"
seasonal_model_path = "your_seasonal_model.pth"
result_path = "your_predicted_y6.csv"  # 예: 250415_Dlinear_y6.csv
trend_data_path = "your_select_feature_trend.csv"
seasonal_data_path = "your_select_feature_seasonal.csv"
output_dir = "your_output_dir"
os.makedirs(output_dir, exist_ok=True)

# --- 데이터 로드 ---
df_y = pd.read_csv(result_path, parse_dates=['date'], index_col='date')
trend_x = pd.read_csv(trend_data_path, parse_dates=['Date'], index_col='Date')
seasonal_x = pd.read_csv(seasonal_data_path, parse_dates=['Date'], index_col='Date')

# --- 정리 ---
for df in [trend_x, seasonal_x]:
    for col in ['Unnamed: 0', 'Y6_1_1_2', 'Y6_2_0_2']:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

trend_x = trend_x.loc[df_y.index]
seasonal_x = seasonal_x.loc[df_y.index]

# ==============================
# 2. 가중치 분석
# ==============================
def extract_weights(model_path, input_size):
    model_cfg = cfg['model_cfg'].copy()
    model_cfg['input_size'] = input_size
    model = Model(**model_cfg)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    w1 = model.fc1.weight.detach().cpu().numpy().flatten()
    w2 = model.fc2.weight.detach().cpu().numpy().flatten()
    return w1, w2

w1_trend, w2_trend = extract_weights(trend_model_path, len(trend_x.columns))
w1_seasonal, w2_seasonal = extract_weights(seasonal_model_path, len(seasonal_x.columns))

# dict로 정리
trend_weights = dict(zip(trend_x.columns, w1_trend + w2_trend))
seasonal_weights = dict(zip(seasonal_x.columns, w1_seasonal + w2_seasonal))

df_weight = pd.DataFrame({'X': list(set(trend_weights.keys()).union(seasonal_weights.keys()))})
df_weight['trend_weight'] = df_weight['X'].map(trend_weights).fillna(0)
df_weight['seasonal_weight'] = df_weight['X'].map(seasonal_weights).fillna(0)
df_weight['total_weight'] = df_weight['trend_weight'] + df_weight['seasonal_weight']
df_weight['abs_total'] = df_weight['total_weight'].abs()
df_weight.sort_values('abs_total', ascending=False).to_csv(f"{output_dir}/dlinear_weight_summary.csv", index=False, encoding='cp949')

# 가중치 시각화
top_n = 20
top_weight_df = df_weight.sort_values('abs_total', ascending=False).head(top_n)
plt.figure(figsize=(10, 6))
plt.barh(top_weight_df['X'], top_weight_df['total_weight'], color='steelblue')
plt.gca().invert_yaxis()
plt.title("Top Variable Weights (fc1 + fc2)")
plt.xlabel("Total Weight")
plt.tight_layout()
plt.savefig(f"{output_dir}/weight_top{top_n}.png", dpi=300)
plt.close()

# ==============================
# 3. 시계열 기여도 분석
# ==============================
def calc_contributions(x_df, weight_dict):
    contrib = pd.DataFrame(index=x_df.index)
    for col in x_df.columns:
        w = weight_dict.get(col, 0)
        contrib[col] = x_df[col] * w
    return contrib

trend_contrib = calc_contributions(trend_x, trend_weights)
seasonal_contrib = calc_contributions(seasonal_x, seasonal_weights)
total_contrib = trend_contrib.add(seasonal_contrib, fill_value=0)
total_contrib['y6_pred'] = df_y['y6']

total_contrib.to_csv(f"{output_dir}/dlinear_contribution_timeseries.csv", encoding='cp949')

# 상위 변수 시각화
top_vars = total_contrib.drop(columns='y6_pred').abs().mean().sort_values(ascending=False).head(10).index

plt.figure(figsize=(14, 6))
for var in top_vars:
    plt.plot(total_contrib.index, total_contrib[var], label=var)
plt.title("Top 10 Variable Contributions Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/top10_contrib_lineplot.png", dpi=300)
plt.close()

# 개별 subplot
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(14, 14), sharex=True)
axes = axes.flatten()
for i, var in enumerate(top_vars):
    axes[i].plot(total_contrib.index, total_contrib[var])
    axes[i].set_title(var)
    axes[i].grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/top10_contrib_subplots.png", dpi=300)
plt.close()

# ==============================
# 4. 변곡점 분석
# ==============================
dy = df_y['y6'].diff().fillna(0)
peaks, _ = find_peaks(np.abs(dy), distance=2, height=dy.std())
key_dates = df_y.index[peaks]

summary = []
for date in key_dates:
    top_at_date = total_contrib.loc[date].drop('y6_pred').abs().sort_values(ascending=False).head(5)
    summary.append({
        'date': date.strftime('%Y-%m'),
        'top_variables': ', '.join(top_at_date.index),
        'values': ', '.join([f"{v:.2f}" for v in top_at_date.values])
    })

df_summary = pd.DataFrame(summary)
df_summary.to_csv(f"{output_dir}/turning_point_summary.csv", index=False, encoding='cp949')

# 변곡점 시각화
plt.figure(figsize=(14, 6))
plt.plot(df_y.index, df_y['y6'], label='Y6 Prediction', linewidth=2)
plt.scatter(df_y.index[peaks], df_y['y6'].iloc[peaks], color='red', label='Turning Points')
plt.title("Y6 Forecast with Turning Points")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/turning_points_plot.png", dpi=300)
plt.close()

print("✅ 전체 분석 완료!")
