import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import os

# 1. 사용자 입력 경로 수정
original_csv_path = "your_input.csv"  # ← 반드시 실제 파일 경로로 바꿔주세요
output_dir = "stl_output_final_strict"
os.makedirs(output_dir, exist_ok=True)

# 2. 설정
start_date = pd.to_datetime("2019-01-01")
stl_period = 12

# 3. 데이터 로드 및 정리
df = pd.read_csv(original_csv_path, encoding='cp949')
df.columns = [col.strip() for col in df.columns]
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df = df.sort_index()

# 4. STL 결과 저장
seasonal_series = {}
trend_series = {}

for col in df.columns:
    series = df[col].copy()
    series = series[series.index >= start_date]

    if series.isna().all() or len(series.dropna()) < stl_period * 2:
        print(f"[스킵] {col} → 유효 데이터 부족")
        continue

    # 내부 보간만 수행하고 앞뒤 NaN 유지
    series = series.interpolate(method='linear', limit_area='inside')
    series = series.dropna()

    if len(series) < stl_period * 2:
        print(f"[스킵] {col} → 보간 후 길이 부족")
        continue

    try:
        stl = STL(series, period=stl_period)
        res = stl.fit()
        seasonal_series[col] = res.seasonal
        trend_series[col] = res.trend
    except Exception as e:
        print(f"[STL 실패] {col}: {e}")

# 5. 병합 (길이 각기 다름, 패딩 없음)
seasonal_df = pd.DataFrame.from_dict(seasonal_series)
trend_df = pd.DataFrame.from_dict(trend_series)

# 6. 저장
seasonal_df.to_csv(f"{output_dir}/stl_seasonal_final.csv", encoding='cp949')
trend_df.to_csv(f"{output_dir}/stl_trend_final.csv", encoding='cp949')

print("✅ STL 분해 최종 완료!")
print(f" - Seasonal 저장: {output_dir}/stl_seasonal_final.csv")
print(f" - Trend 저장   : {output_dir}/stl_trend_final.csv")
