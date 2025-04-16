import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model.net import DLinear_with_emb as Model
from config import cfg

# -------- 사용자 설정 --------
dir_out = '250415_Y6_trend_Dlinear'  # 학습 결과 폴더
x_names = [...]  # final_x_columns 와 동일하게 사용한 X 인자 이름 리스트
input_size = len(x_names)
output_path = os.path.join(dir_out, "dlinear_weights_summary.csv")
# -----------------------------

# 1. 모델 설정
model_cfg = cfg['model_cfg']
model_cfg['input_size'] = input_size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 가중치 수집
all_weights = []

for iid in range(100):
    model_path = os.path.join(dir_out, f"trend_Dlinear_{iid}.pth")
    if not os.path.exists(model_path):
        print(f"[경고] 모델 없음: {model_path}")
        continue

    model = Model(**model_cfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    weights = model.linear.weight.detach().cpu().numpy().flatten()
    all_weights.append(weights)

# 3. 평균 및 통계 계산
all_weights = np.array(all_weights)  # (100, num_x)
avg_weights = np.mean(all_weights, axis=0)
std_weights = np.std(all_weights, axis=0)

df_summary = pd.DataFrame({
    'X_variable': x_names,
    'avg_weight': avg_weights,
    'std_weight': std_weights,
    'abs_avg_weight': np.abs(avg_weights)
}).sort_values(by='abs_avg_weight', ascending=False)

# 4. 상위 영향력 X인자 시각화
top_n = 20
top_df = df_summary.head(top_n)

plt.figure(figsize=(10, 6))
plt.barh(top_df['X_variable'], top_df['avg_weight'])
plt.gca().invert_yaxis()
plt.title(f'Top {top_n} Influential X Variables (DLinear)')
plt.xlabel("Average Weight (across 100 models)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(dir_out, "dlinear_weight_top20.png"))
plt.show()

# 5. 저장
df_summary.to_csv(output_path, index=False, encoding='cp949')
print(f"✅ 분석 완료: 요약 파일 저장 → {output_path}")
