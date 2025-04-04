import pandas as pd
from statsmodels.tsa.seasonal import STL
import os

# STL 분해 함수
def decompose_and_save(df, period=12, output_dir="stl_components"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    seasonal_df = pd.DataFrame({'Date': df['Date']})
    trend_df = pd.DataFrame({'Date': df['Date']})

    for col in df.columns:
        if col == 'Date':
            continue

        series = df[['Date', col]].dropna()
        series = series[series[col] != 0]
        
        if len(series) < period * 2:  # 데이터가 너무 짧으면 스킵
            continue

        # 시계열 정렬 및 인덱스 설정
        ts = series.set_index('Date')[col].asfreq('MS')  # 월별 시작일로 강제
        ts = ts.interpolate(method='linear')  # 결측 보간

        try:
            stl = STL(ts, period=period, robust=True)
            result = stl.fit()

            # 결과 저장
            seasonal = result.seasonal.reindex(df['Date'])
            trend = result.trend.reindex(df['Date'])

            seasonal_df[col] = seasonal.values
            trend_df[col] = trend.values

        except Exception as e:
            print(f"Error processing {col}: {e}")

    # 저장
    seasonal_df.to_csv(os.path.join(output_dir, 'seasonal.csv'), index=False)
    trend_df.to_csv(os.path.join(output_dir, 'trend.csv'), index=False)

    print("STL 분해 완료: seasonal.csv, trend.csv 저장됨.")

# 사용 예시
# df = pd.read_csv('your_data.csv', parse_dates=['Date'])
# decompose_and_save(df)
