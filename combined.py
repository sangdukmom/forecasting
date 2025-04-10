import pandas as pd

# [1] 경로 설정
sarima_seasonal_path = "경로/250410_sarima_selected_seasonal.csv"
sarima_trend_path = "경로/250410_sarima_selected_trend.csv"
stl_seasonal_path = "경로/stl_seasonal_final.csv"
stl_trend_path = "경로/stl_trend_final.csv"
skipped_seasonal_path = "경로/skipped_variables_seasonal.csv"
skipped_trend_path = "경로/skipped_variables_trend.csv"

# [2] 데이터 로딩
sarima_seasonal = pd.read_csv(sarima_seasonal_path, encoding='cp949')
sarima_trend = pd.read_csv(sarima_trend_path, encoding='cp949')
stl_seasonal = pd.read_csv(stl_seasonal_path, encoding='cp949')
stl_trend = pd.read_csv(stl_trend_path, encoding='cp949')

for df in [sarima_seasonal, sarima_trend, stl_seasonal, stl_trend]:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

# [3] SARIMA 생략된 변수명 로딩
with open(skipped_seasonal_path, 'r', encoding='utf-8') as f:
    skipped_seasonal_vars = [line.strip() for line in f if line.strip()]

with open(skipped_trend_path, 'r', encoding='utf-8') as f:
    skipped_trend_vars = [line.strip() for line in f if line.strip()]

# [4] STL 결과 중 생략 변수 추출
stl_skipped_seasonal = stl_seasonal[skipped_seasonal_vars]
stl_skipped_trend = stl_trend[skipped_trend_vars]

# [5] SARIMA + STL 병합
combined_seasonal = pd.concat([sarima_seasonal, stl_skipped_seasonal], axis=1)
combined_trend = pd.concat([sarima_trend, stl_skipped_trend], axis=1)

# [6] 저장
combined_seasonal.to_csv("250410_combined_seasonal.csv", encoding='cp949')
combined_trend.to_csv("250410_combined_trend.csv", encoding='cp949')

print("✅ 통합 완료! → 250410_combined_seasonal.csv, 250410_combined_trend.csv 생성됨.")
