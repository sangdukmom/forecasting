import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# 설정
INPUT_FILES = {
    'seasonal': 'seasonal.csv',
    'trend': 'trend.csv'
}
OUTPUT_DIR = 'sarima_forecast_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SARIMA 설정
SARIMA_ORDERS = [(p, d, q) for p in range(3) for d in range(3) for q in range(3)]
TARGET_END_DATE = pd.to_datetime('2025-12-01')

# 예측 함수
def forecast_sarimax(df, column, order, end_date):
    series = df[['date', column]].dropna()
    series = series[series[column] != 0]

    if len(series) < 24:  # 데이터 너무 짧으면 skip
        return None

    ts = series.set_index('date')[column].asfreq('MS')
    ts = ts.interpolate('linear')

    if ts.index[-1] >= end_date:
        return None  # 이미 예측 대상 구간 포함된 경우

    try:
        model = SARIMAX(ts, order=order, seasonal_order=order + (12,))
        results = model.fit(disp=False)

        months_to_predict = ((end_date.year - ts.index[-1].year) * 12 +
                             (end_date.month - ts.index[-1].month))
        forecast = results.get_prediction(start=len(ts),
                                          end=len(ts) + months_to_predict)
        forecast_index = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1),
                                       periods=months_to_predict + 1, freq='MS')
        forecast_df = pd.DataFrame({column: forecast.predicted_mean, 'date': forecast_index})
        return forecast_df
    except:
        return None

# 전체 처리 함수
def process_and_forecast(input_file, label):
    df_raw = pd.read_csv(input_file)
    df_raw['date'] = pd.to_datetime(df_raw['date'])

    var_columns = [col for col in df_raw.columns if col != 'date']
    df_merged = df_raw[['date']].copy()

    for col in tqdm(var_columns, desc=f'Processing {label}'):
        df_data = df_raw[['date', col]].copy()

        fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=True)
        plt.suptitle(f'SARIMA Forecast: {col}')

        for i, (p, d, q) in enumerate(SARIMA_ORDERS):
            ax = axes[i // 3][i % 3]
            ax.plot(df_data['date'], df_data[col], label='Observed')

            forecast_df = forecast_sarimax(df_data, col, (p, d, q), TARGET_END_DATE)

            if forecast_df is not None:
                combined = pd.concat([df_data, forecast_df], ignore_index=True)
                ax.plot(combined['date'], combined[col], label=f'({p},{d},{q})')
                combined = combined.rename(columns={col: f'{col}_{p}_{d}_{q}'})
                df_merged = pd.merge(df_merged, combined[['date', f'{col}_{p}_{d}_{q}']],
                                     on='date', how='outer')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/sarima_{label}_{col}.png", dpi=300)

        for ax in axes.flatten():
            ax.set_xlim(pd.to_datetime('2022-01-01'), TARGET_END_DATE)

        plt.savefig(f"{OUTPUT_DIR}/sarima_{label}_{col}_detailed.png", dpi=300)
        plt.close()

    # 저장
    df_merged.to_csv(f"{OUTPUT_DIR}/sarima_{label}.csv", index=False, encoding='utf-8-sig')
    print(f"저장 완료: sarima_{label}.csv")

# 실행
process_and_forecast(INPUT_FILES['seasonal'], 'seasonal')
process_and_forecast(INPUT_FILES['trend'], 'trend')
