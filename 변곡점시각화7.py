# 예측값에서 1차 차분
y = df_y['y6']
dy = y.diff().fillna(0)

# 부호 변화 탐지: sign 전환 지점 (ex. + → - or - → +)
sign_change = (np.sign(dy).shift(-1) != np.sign(dy)) & (np.sign(dy) != 0)

# 변화량이 일정 threshold 이상인 경우만
threshold = 150  # 튜닝 가능
significant = dy.abs() > threshold

# 두 조건 모두 만족하는 변곡점
turning_points = df_y.index[sign_change & significant]

# 시각화
plt.figure(figsize=(14,6))
plt.plot(df_y.index, y, label='y6 prediction', linewidth=2)
plt.scatter(turning_points, y.loc[turning_points], color='red', label='Turning Point', zorder=5)
plt.title("Turning Point Detection (Sign Change + Threshold)")
plt.xlabel("Date")
plt.ylabel("y6")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/turning_points_improved.png", dpi=300)
plt.show()
