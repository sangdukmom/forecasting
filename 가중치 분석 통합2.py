# ✅ fc1 + fc2 통합 분석

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.net import DLinear_with_emb as Model
from config import cfg

# ---- 사용자 설정 ----
model_path = '경로/최종모델.pth'
x_names = [...]  # 사용된 X인자 이름
output_dir = '경로/저장폴더'
top_n = 20
# --------------------

model_cfg = cfg['model_cfg']
model_cfg['input_size'] = len(x_names)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model(**model_cfg).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Trend / Seasonal 가중치 추출
fc1_weights = model.fc1.weight.detach().cpu().numpy().flatten()
fc2_weights = model.fc2.weight.detach().cpu().numpy().flatten()

# 총합 가중치
total_weights = fc1_weights + fc2_weights

df_weights = pd.DataFrame({
    'X_variable': x_names,
    'trend_weight': fc1_weights,
    'seasonal_weight': fc2_weights,
    'total_weight': total_weights,
    'abs_total_weight': np.abs(total_weights)
}).sort_values(by='abs_total_weight', ascending=False)

df_weights.to_csv(f"{output_dir}/dlinear_total_weight_summary.csv", index=False, encoding='cp949')

# 시각화
top_df = df_weights.head(top_n)
plt.figure(figsize=(10, 6))
plt.barh(top_df['X_variable'], top_df['total_weight'], color='slateblue')
plt.gca().invert_yaxis()
plt.xlabel("Trend + Seasonal Weight")
plt.title("Top X Variables by Total Weight")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/dlinear_total_weight_top{top_n}.png", dpi=300)
plt.show()

print("✅ 총합 가중치 분석 완료!")
