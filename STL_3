import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import os

# 1. 데이터 경로 수정하세요
original_csv_path = "your_input.csv"  # 실제 데이터 경로로 변경
output_dir = "stl_output_varlen_from_2019"
os.makedirs(output_dir, exist_ok=True)

# 2. 설정
start_date = pd.to_datetime("2019-01-01")
stl_period = 12

# 3. 데이터 로드
df = pd.read_csv(original_csv_path, encoding='cp949')
df.columns = [col.strip() for col in df.columns]
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df = df.sort_index()

# 4. STL 분해 결과 저장용
seasonal_dict = {}
trend_dict = {}

# 5. 컬럼별 STL 수행
for col in df.columns:
    series = df[col].copy()
    series = series[series.index >= start_date]

    if series.isna().all() or len(series.dropna()) < stl_period * 2:
        print(f"[스킵] {col} → 유효 구간 부족")
        continue

    series = series.interpolate(method='linear', limit_area='inside').bfill().ffill()

    try:
        stl = STL(series, period=stl_period)
        res = stl.fit()
        seasonal_dict[col] = res.seasonal
        trend_dict[col] = res.trend
    except Exception as e:
        print(f"[STL 실패] {col}: {e}")

# 6. 결과 병합 및 저장
seasonal_df = pd.DataFrame(seasonal_dict)
trend_df = pd.DataFrame(trend_dict)

seasonal_df.to_csv(os.path.join(output_dir, "stl_seasonal_from_2019.csv"), encoding='cp949')
trend_df.to_csv(os.path.join(output_dir, "stl_trend_from_2019.csv"), encoding='cp949')

print("STL 분해 완료!")
