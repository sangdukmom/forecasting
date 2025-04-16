import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.net import DLinear_with_emb as Model
from config import cfg

# -------- 사용자 설정 --------
trend_model_path = "경로/trend_Dlinear_*.pth"
seasonal_model_path = "경로/seasonal_Dlinear_*.pth"

trend_x_names = [...]     # trend 모델의 X 인자 리스트
seasonal_x_names = [...]  # seasonal 모델의 X 인자 리스트

output_dir = "경로/저장폴더"
top_n = 20  # 시각화할 변수 수
# -----------------------------

# 모델 config
def load_model(path, input_size):
    model_cfg = cfg['model_cfg'].copy()
    model_cfg['input_size'] = input_size
    model = Model(**model_cfg)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

# 1. Load both models
model_trend = load_model(trend_model_path, len(trend_x_names))
model_seasonal = load_model(seasonal_model_path, len(seasonal_x_names))

# 2. Extract weights
w_trend = model_trend.linear.weight.detach().cpu().numpy().flatten()
w_seasonal = model_seasonal.linear.weight.detach().cpu().numpy().flatten()

# 3. Build weight DataFrames
df_trend = pd.DataFrame({'X': trend_x_names, 'trend_weight': w_trend})
df_seasonal = pd.DataFrame({'X': seasonal_x_names, 'seasonal_weight': w_seasonal})

# 4. Merge & fill missing with 0
df_merged = pd.merge(df_trend, df_seasonal, on='X', how='outer').fillna(0)
df_merged['total_weight'] = df_merged['trend_weight'] + df_merged['seasonal_weight']
df_merged['abs_total_weight'] = np.abs(df_merged['total_weight'])

# 5. Sort by importance
df_sorted = df_merged.sort_values(by='abs_total_weight', ascending=False)

# 6. Save result
df_sorted.to_csv(f"{output_dir}/dlinear_total_weight_summary.csv", index=False, encoding='cp949')

# 7. Plot top variables
top_df = df_sorted.head(top_n)
plt.figure(figsize=(10, 6))
plt.barh(top_df['X'], top_df['total_weight'], color='teal')
plt.gca().invert_yaxis()
plt.xlabel("Total Weight (Trend + Seasonal)")
plt.title(f"Top {top_n} Influential X Variables (DLinear Total)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/dlinear_total_weight_top{top_n}.png", dpi=300)
plt.show()

print("✅ 통합 분석 완료! 결과 저장됨.")
