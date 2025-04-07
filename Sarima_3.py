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

SARIMA_ORDERS = [(p, d, q) for p in range(3) for d in range(3) for q in range(3)]
TARGET_END_DATE = pd.to_datetime('2025-12-01')

# SARIMA 예측 함수
def forecast_sarimax(df, column, order, end_date):
    series = df[['Date', column]].dropna()
    series = series[series[column] != 0]

    if len(series) < 24:
        return None

    ts = series.set_index('Date')[column].asfreq('MS')
    ts = ts.interpolate('linear')

    if ts.index[-1] >= end_date:
        return None

    try:
        model = SARIMAX(ts, order=order, seasonal_order=order + (12,))
        results = model.fit(disp=False)

        months_to_predict = ((end_date.year - ts.index[-1].year) * 12 +
                             (end_date.month - ts.index[-1].month))

        forecast = results.get_prediction(start=len(ts),
                                          end=len(ts) + months_to_predict)

        forecast_values = forecast.predicted_mean
        if hasattr(forecast_values, 'ndim') and forecast_values.ndim > 1:
            forecast_values = forecast_values.values.ravel()

        forecast_index = pd.date_range(
            start=ts.index[-1] + pd.DateOffset(months=1),
            periods=months_to_predict + 1, freq='MS'
        )
        if forecast_index.ndim > 1:
            forecast_index = forecast_index.ravel()

        forecast_df = pd.DataFrame({
            'Date': forecast_index,
            f'{column}': forecast_values
        })

        return forecast_df

    except Exception as e:
        print(f"SARIMA 예측 실패: {column}, order={order}, error={e}")
        return None

# 전체 처리 함수
def process_and_forecast(input_file, label):
    df_raw = pd.read_csv(input_file)
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])

    var_columns = [col for col in df_raw.columns if col != 'Date']

    for col in tqdm(var_columns, desc=f'Processing {label}'):
        df_data = df_raw[['Date', col]].copy()

        fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=True)
        plt.suptitle(f'SARIMA Forecast: {col}')

        for i, (p, d, q) in enumerate(SARIMA_ORDERS):
            ax = axes[i // 3][i % 3]
            ax.plot(df_data['Date'], df_data[col], label='Observed')

            forecast_df = forecast_sarimax(df_data, col, (p, d, q), TARGET_END_DATE)

            if forecast_df is not None:
                ax.plot(forecast_df['Date'], forecast_df[col], label=f'({p},{d},{q})')

                # CSV 저장 (변수명 + p_d_q)
                filename = f"{OUTPUT_DIR}/sarima_{label}_{col}_{p}_{d}_{q}.csv"
                forecast_df.to_csv(filename, index=False, encoding='utf-8-sig')

            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/sarima_{label}_{col}.png", dpi=300)

        for ax in axes.flatten():
            ax.set_xlim(pd.to_datetime('2022-01-01'), TARGET_END_DATE)

        plt.savefig(f"{OUTPUT_DIR}/sarima_{label}_{col}_detailed.png", dpi=300)
        plt.close()

    print(f"[{label}] 처리 완료. 결과 파일은 '{OUTPUT_DIR}' 폴더에 저장됨.")

# 실행
process_and_forecast(INPUT_FILES['seasonal'], 'seasonal')
process_and_forecast(INPUT_FILES['trend'], 'trend')
