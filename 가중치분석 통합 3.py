import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.net import DLinear_with_emb as Model
from config import cfg

# -----------------------------
# 사용자 설정
trend_model_path = '경로/트렌드_모델.pth'
seasonal_model_path = '경로/시즈널_모델.pth'
trend_x_names = [...]     # trend 예측 모델에 사용된 X인자 리스트
seasonal_x_names = [...]  # seasonal 예측 모델에 사용된 X인자 리스트
output_dir = '경로/저장폴더'
top_n = 20
# -----------------------------

def load_fc1_fc2_weights(path, input_size):
    model_cfg = cfg['model_cfg'].copy()
    model_cfg['input_size'] = input_size
    model = Model(**model_cfg)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    w_fc1 = model.fc1.weight.detach().cpu().numpy().flatten()
    w_fc2 = model.fc2.weight.detach().cpu().numpy().flatten()
    return w_fc1 + w_fc2

# 1. 각각의 모델에서 fc1 + fc2 합산 가중치 추출
w_trend = load_fc1_fc2_weights(trend_model_path, len(trend_x_names))
w_seasonal = load_fc1_fc2_weights(seasonal_model_path, len(seasonal_x_names))

# 2. 데이터프레임 생성
df_trend = pd.DataFrame({'X': trend_x_names, 'trend_weight': w_trend})
df_seasonal = pd.DataFrame({'X': seasonal_x_names, 'seasonal_weight': w_seasonal})

# 3. 통합: 동일 X변수 기준으로 합산
df_merged = pd.merge(df_trend, df_seasonal, on='X', how='outer').fillna(0)
df_merged['total_weight'] = df_merged['trend_weight'] + df_merged['seasonal_weight']
df_merged['abs_total_weight'] = np.abs(df_merged['total_weight'])

# 4. 저장 및 시각화
df_sorted = df_merged.sort_values(by='abs_total_weight', ascending=False)
df_sorted.to_csv(f"{output_dir}/dlinear_total_weight_summary.csv", index=False, encoding='cp949')

# 5. Plot top N
top_df = df_sorted.head(top_n)
plt.figure(figsize=(10, 6))
plt.barh(top_df['X'], top_df['total_weight'], color='teal')
plt.gca().invert_yaxis()
plt.xlabel("Trend + Seasonal Total Weight")
plt.title("Top X Variables by Total DLinear Weight")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/dlinear_total_weight_top{top_n}.png", dpi=300)
plt.show()

print("✅ 두 모델의 fc1 + fc2 합산 분석 완료!")
