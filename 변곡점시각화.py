from scipy.signal import find_peaks

# === 예측값 불러오기 ===
y = df_y['y6']
y_2025 = y['2025-01':'2025-12']  # 2025년 구간만 분석

# === 1차 차분으로 변화량 계산 ===
dy = y_2025.diff().fillna(0)

# === 변화 방향: 증가→감소 / 감소→증가 위치 찾기 ===
sign_change = np.sign(dy).diff().fillna(0)
turning_points = sign_change[sign_change != 0].index.tolist()

# === 영향력 높은 변수 찾기 ===
summary = []
for date in turning_points:
    if date not in total_contrib.index:
        continue
    contrib_on_date = total_contrib.loc[date].drop('y6_pred').abs().sort_values(ascending=False)
    top_vars = contrib_on_date.head(5)
    summary.append({
        'date': date.strftime('%Y-%m'),
        'top_variables': ', '.join(top_vars.index),
        'values': ', '.join([f"{v:.2f}" for v in top_vars.values])
    })

df_turning = pd.DataFrame(summary)
df_turning.to_csv(f"{output_dir}/turning_point_analysis_2025.csv", index=False, encoding='cp949')

# === 시각화 ===
plt.figure(figsize=(14, 5))
plt.plot(y_2025.index, y_2025.values, label='Y6 Forecast', color='black')
plt.scatter(turning_points, y_2025.loc[turning_points], color='red', marker='o', label='Turning Point')
plt.title("Turning Points in 2025 – DLinear Forecast")
plt.xlabel("Date")
plt.ylabel("Y6 Forecast")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/turning_points_2025.png", dpi=300)
plt.show()
