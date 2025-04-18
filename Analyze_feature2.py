# =============================================
# 1. 준비: 경로 설정 및 데이터 로드
# =============================================
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model.net import DLinear_with_emb as Model
from config import cfg
from scipy.signal import find_peaks

# ---- 사용자 경로 설정 ----
date = "250415"
target = "y6"

trend_model_path = "trend_model.pth"
seasonal_model_path = "seasonal_model.pth"

result_path = f"{date}_Dlinear_{target}.csv"  # 예측값
trend_data_path = "select_feature_trend.csv"
seasonal_data_path = "select_feature_seasonal.csv"

output_dir = f"{date}_{target}_final_analysis"
os.makedirs(output_dir, exist_ok=True)

# ---- 데이터 로드 ----
df_y = pd.read_csv(result_path, parse_dates=['date'], index_col='date')
trend_x = pd.read_csv(trend_data_path, parse_dates=['Date'], index_col='Date')
seasonal_x = pd.read_csv(seasonal_data_path, parse_dates=['Date'], index_col='Date')

# ===== 불필요 컬럼 제거 (필요시 수정) =====
for df in [trend_x, seasonal_x]:
    for col in ['Unnamed: 0', 'Y6_2_0_2', 'Y6_1_1_2']:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

trend_x = trend_x.loc[df_y.index]
seasonal_x = seasonal_x.loc[df_y.index]

# =============================================
# 2. 가중치 분석
# =============================================
def extract_weights(model_path, input_size):
    model_cfg = cfg['model_cfg'].copy()
    model_cfg['input_size'] = input_size
    model = Model(**model_cfg)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model.fc1.weight.detach().cpu().numpy().flatten(), model.fc2.weight.detach().cpu().numpy().flatten()

w_fc1_trend, w_fc2_trend = extract_weights(trend_model_path, len(trend_x.columns))
w_fc1_seasonal, w_fc2_seasonal = extract_weights(seasonal_model_path, len(seasonal_x.columns))

# dict로 정리
trend_weights = dict(zip(trend_x.columns, w_fc1_trend + w_fc2_trend))
seasonal_weights = dict(zip(seasonal_x.columns, w_fc1_seasonal + w_fc2_seasonal))

# 저장
df_weight = pd.DataFrame({
    'X': list(set(trend_weights.keys()).union(seasonal_weights.keys()))
})
df_weight['trend_weight'] = df_weight['X'].map(trend_weights).fillna(0)
df_weight['seasonal_weight'] = df_weight['X'].map(seasonal_weights).fillna(0)
df_weight['total_weight'] = df_weight['trend_weight'] + df_weight['seasonal_weight']
df_weight['abs_total'] = df_weight['total_weight'].abs()
df_weight.sort_values('abs_total', ascending=False).to_csv(f"{output_dir}/dlinear_weight_summary.csv", index=False, encoding='cp949')

# =============================================
# 3. 시계열 기여도 분석
# =============================================
def calc_contributions(x_df, weight_dict):
    contrib = pd.DataFrame(index=x_df.index)
    for col in x_df.columns:
        w = weight_dict.get(col, 0)
        contrib[col] = x_df[col] * w
    return contrib

trend_contrib = calc_contributions(trend_x, trend_weights)
seasonal_contrib = calc_contributions(seasonal_x, seasonal_weights)
total_contrib = trend_contrib.add(seasonal_contrib, fill_value=0)

# 예측값 추가
total_contrib['y6'] = df_y['y6']
total_contrib.to_csv(f"{output_dir}/dlinear_contribution_timeseries.csv", encoding='cp949')

# 시각화
top_vars = total_contrib.drop(columns='y6').abs().mean().sort_values(ascending=False).head(10).index
plt.figure(figsize=(14,6))
for var in top_vars:
    plt.plot(total_contrib.index, total_contrib[var], label=var)
plt.title("Top 10 Variable Contributions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/top10_variable_contributions.png", dpi=300)
plt.show()

# =============================================
# 4. 변곡점 자동 탐지 및 분석
# =============================================
# 예측값의 1차 차분
dy = df_y['y6'].diff().fillna(0)

# find_peaks 로 변화량의 peak 탐지
peaks, _ = find_peaks(np.abs(dy), distance=2, height=dy.std())
key_dates = df_y.index[peaks]

# 상위 기여 변수 탐색
summary_list = []
for date in key_dates:
    top_vars = total_contrib.loc[date].drop('y6').abs().sort_values(ascending=False).head(5)
    summary_list.append({
        'date': date.strftime('%Y-%m'),
        'top_variables': ', '.join(top_vars.index),
        'values': ', '.join([f"{v:.2f}" for v in top_vars.values])
    })

df_summary = pd.DataFrame(summary_list)
df_summary.to_csv(f"{output_dir}/turning_point_analysis.csv", index=False, encoding='cp949')
print("✅ 변곡점 분석 결과 저장 완료!")
