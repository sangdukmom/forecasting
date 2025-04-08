import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import os

# 1. 사용자 입력: 경로만 수정하세요
original_csv_path = "your_input.csv"  # STL 분해할 원본 데이터 경로
output_dir = "stl_output_2019_2024"
os.makedirs(output_dir, exist_ok=True)

# 2. 설정
start_date = "2019-01-01"
end_date = "2024-12-01"
stl_period = 12  # 월별 주기

# 3. 데이터 로드 및 정제
df = pd.read_csv(original_csv_path, encoding='cp949')
df.columns = [col.strip() for col in df.columns]
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df = df.sort_index()
df = df.loc[start_date:end_date]

# 4. 결과 저장용 데이터프레임 생성
seasonal_df = pd.DataFrame(index=df.index)
trend_df = pd.DataFrame(index=df.index)

# 5. STL 분해
for col in df.columns:
    series = df[col].copy()
    if series.isna().all():
        print(f"[스킵] {col} → 모두 결측치")
        continue

    # 결측치 보간
    series = series.interpolate(method='linear', limit_area='inside').bfill().ffill()

    try:
        stl = STL(series, period=stl_period)
        res = stl.fit()
        seasonal_df[col] = res.seasonal
        trend_df[col] = res.trend
    except Exception as e:
        print(f"[STL 실패] {col}: {e}")

# 6. 파일 저장
seasonal_path = os.path.join(output_dir, "stl_seasonal_2019_2024.csv")
trend_path = os.path.join(output_dir, "stl_trend_2019_2024.csv")

seasonal_df.to_csv(seasonal_path, encoding='cp949')
trend_df.to_csv(trend_path, encoding='cp949')

print("STL 분해 완료:")
print(f" - Seasonal 저장: {seasonal_path}")
print(f" - Trend 저장   : {trend_path}")
