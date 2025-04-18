import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import cfg

# ============ 사용자 설정 ============
output_dir = "output_path"  # 저장 폴더
target = "y6"
threshold = 50  # 변화량 임계값
top_k = 5       # 기여도 상위 변수 개수

# ============ 데이터 로드 ============
df_y = pd.read_csv("250415_Dlinear_y6.csv", parse_dates=["date"], index_col="date")
df_contrib = pd.read_csv(f"{output_dir}/dlinear_contribution_timeseries.csv", encoding='cp949', index_col='date', parse_dates=True)

# ============ 변곡점 탐지 ============
y = df_y['y6']
dy = y.diff().fillna(0)

sign_change = (np.sign(dy).shift(-1) != np.sign(dy)) & (np.sign(dy) != 0)
significant = dy.abs() > threshold

turning_points = df_y.index[sign_change & significant]
turning_points_2025 = [d for d in turning_points if d.year == 2025]

# ============ 변곡점 시각화 ============
plt.figure(figsize=(14, 6))
plt.plot(df_y.index, y, label='y6 prediction', linewidth=2)
plt.scatter(turning_points_2025, y.loc[turning_points_2025], color='red', label='Turning Point', zorder=5)
plt.title("Turning Point Detection (Sign Change + Threshold)")
plt.xlabel("Date")
plt.ylabel("y6")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/turning_points_improved.png", dpi=300)
plt.show()

# ============ 각 변곡점에서 주요 변수 기여도 추출 ============
summary = []

for date in turning_points_2025:
    contrib_at_date = df_contrib.loc[date].drop('y6')  # 'y6' 제외
    top_vars = contrib_at_date.abs().sort_values(ascending=False).head(top_k)
    
    summary.append({
        'date': date.strftime('%Y-%m'),
        'top_variables': ', '.join(top_vars.index),
        'values': ', '.join([f"{v:.2f}" for v in top_vars.values])
    })

df_summary = pd.DataFrame(summary)
df_summary.to_csv(f"{output_dir}/turning_point_contribution_summary.csv", index=False, encoding='cp949')

# ============ 기여도 시각화 ============
for date in turning_points_2025:
    contrib_at_date = df_contrib.loc[date].drop('y6')
    top_vars = contrib_at_date.abs().sort_values(ascending=False).head(top_k)

    plt.figure(figsize=(10, 5))
    top_vars.loc[top_vars.index].plot(kind='barh', color='skyblue')
    plt.title(f"Top {top_k} Contributions at Turning Point: {date.strftime('%Y-%m')}")
    plt.xlabel("Contribution")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/turning_point_contrib_{date.strftime('%Y-%m')}.png", dpi=300)
    plt.close()

print("✅ 변곡점 분석 및 시각화 완료")
