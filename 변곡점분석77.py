import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ========== 사용자 설정 ==========
output_dir = "output_dir"  # 경로 수정 가능
os.makedirs(output_dir, exist_ok=True)

# 예측값과 기여도 파일 로드
df_y = pd.read_csv("250415_Dlinear_y6.csv", parse_dates=["date"], index_col="date")
df_contrib = pd.read_csv(f"{output_dir}/dlinear_contribution_timeseries.csv", encoding='cp949', parse_dates=True, index_col='date')

# ========== 1. 변곡점 탐지 (2025년) ==========
df_y_2025 = df_y.loc["2025"]
dy = df_y_2025['y6'].diff().fillna(0)

# 변화량 기준으로 피크 탐지
peaks, _ = find_peaks(np.abs(dy), distance=1, height=dy.std())
key_dates = df_y_2025.index[peaks]

# ========== 2. 주요 변수 요약 저장 ==========
summary_list = []

for date in key_dates:
    row = df_contrib.loc[date].drop('y6')
    top_vars = row.abs().sort_values(ascending=False).head(5)
    
    summary_list.append({
        "date": date.strftime("%Y-%m"),
        "top_variables": ', '.join(top_vars.index),
        "contributions": ', '.join([f"{v:.2f}" for v in top_vars.values])
    })

df_summary = pd.DataFrame(summary_list)
df_summary.to_csv(f"{output_dir}/turning_point_summary_2025.csv", index=False, encoding='cp949')
print("✅ 변곡점 요약 CSV 저장 완료")

# ========== 3. 기여도 바 차트 시각화 ==========
def plot_contributions_at_turning_points(df_contrib, turning_dates, top_n=5, output_dir=".", prefix="tp_contrib"):
    for date in turning_dates:
        if date not in df_contrib.index:
            continue
        
        row = df_contrib.loc[date].drop('y6')
        top_vars = row.abs().sort_values(ascending=False).head(top_n)
        contrib_vals = row[top_vars.index]

        plt.figure(figsize=(8, 5))
        contrib_vals.loc[::-1].plot(
            kind='barh',
            color=['green' if v > 0 else 'red' for v in contrib_vals.loc[::-1]],
            edgecolor='black'
        )
        plt.title(f"Variable Contributions on {date.strftime('%Y-%m')}")
        plt.xlabel("Contribution to y6")
        plt.grid(True)
        plt.tight_layout()

        fname = f"{prefix}_{date.strftime('%Y%m')}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=300)
        plt.close()

    print(f"✅ 총 {len(turning_dates)}개 변곡점 기여도 그래프 저장 완료")

plot_contributions_at_turning_points(
    df_contrib=df_contrib,
    turning_dates=key_dates,
    top_n=5,
    output_dir=output_dir,
    prefix="turning_point_contrib"
)

# ========== 4. 변곡점 라인 시각화 ==========
plt.figure(figsize=(12, 6))
plt.plot(df_y.index, df_y['y6'], label='Y6 Forecast', linewidth=2)
plt.scatter(key_dates, df_y.loc[key_dates]['y6'], color='red', label='Turning Points', zorder=5)
for date in key_dates:
    plt.axvline(date, color='red', linestyle='--', alpha=0.4)

plt.title("Y6 Forecast with Detected Turning Points (2025)")
plt.xlabel("Date")
plt.ylabel("Y6 Forecast")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/turning_points_lineplot.png", dpi=300)
plt.show()
