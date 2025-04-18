from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# 예측값에서 2025년 구간만 추출
y_2025 = total_contrib.loc['2025-01':'2025-12', 'y6_pred']

# 1차 도함수 계산 (차분)
dy = y_2025.diff().fillna(0)
dy_prev = dy.shift(1)

# 부호 변화 감지: 상승 → 하강 또는 하강 → 상승
sign_change = (dy * dy_prev < 0)

# 임계값 기준 변화량 필터링 (노이즈 제거)
threshold = dy.std() * 0.3  # 민감도 조절 가능
strong_change = dy.abs() > threshold

# 최종 변곡점 시점 추출
turning_points = dy[sign_change & strong_change].index

# 변곡점마다 주요 X인자 추출
summary_list = []
for date in turning_points:
    top_vars = total_contrib.loc[date].drop('y6_pred').abs().sort_values(ascending=False).head(5)
    summary_list.append({
        'date': date.strftime('%Y-%m'),
        'top_variables': ', '.join(top_vars.index),
        'contributions': ', '.join([f"{total_contrib.loc[date, var]:.2f}" for var in top_vars.index])
    })

df_turning_summary = pd.DataFrame(summary_list)

# 저장
df_turning_summary.to_csv(f"{output_dir}/turning_point_analysis_2025.csv", index=False, encoding='cp949')

# 시각화: 변곡점 위치 표시
plt.figure(figsize=(14, 6))
plt.plot(y_2025.index, y_2025.values, label='Predicted Y6', color='black')
plt.scatter(turning_points, y_2025.loc[turning_points], color='red', label='Turning Point', zorder=5)
plt.title("Y6 Predicted Value with Turning Points (2025)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/turning_points_2025.png", dpi=300)
plt.show()
