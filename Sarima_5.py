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

# SARIMA 예측 및 AIC 평가 함수
def fit_sarimax(ts, order):
    try:
        model = SARIMAX(ts, order=order, seasonal_order=order + (12,))
        result = model.fit(disp=False)
        return result
    except:
        return None

# 전체 처리 함수
def process_and_forecast(input_file, label):
    df_raw = pd.read_csv(input_file)
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])

    var_columns = [col for col in df_raw.columns if col != 'Date']
    df_best_all = pd.DataFrame()

    for col in tqdm(var_columns, desc=f'Processing {label}'):
        df_data = df_raw[['Date', col]].dropna()
        df_data = df_data[df_data[col] != 0]
        if len(df_data) < 24:
            continue

        ts = df_data.set_index('Date')[col].asfreq('MS')
        ts = ts.interpolate('linear')

        best_aic = float('inf')
        best_result = None
        best_order = None
        best_forecast_df = None

        fig, axes = plt.subplots(9, 3, figsize=(18, 30), sharex=True)
        axes = axes.flatten()
        plt.suptitle(f'SARIMA Forecast: {col}', fontsize=16)

        for i, (p, d, q) in enumerate(SARIMA_ORDERS):
            ax = axes[i]
            ax.plot(ts.index, ts.values, label='Observed')

            result = fit_sarimax(ts, (p, d, q))
            if result is None:
                ax.set_title(f'({p},{d},{q}) - FAIL')
                continue

            aic = result.aic
            forecast_steps = ((TARGET_END_DATE.year - ts.index[-1].year) * 12 +
                              (TARGET_END_DATE.month - ts.index[-1].month))
            if forecast_steps <= 0:
                continue

            forecast = result.get_prediction(start=len(ts), end=len(ts) + forecast_steps)
            forecast_index = pd.date_range(
                start=ts.index[-1] + pd.DateOffset(months=1),
                periods=forecast_steps + 1,
                freq='MS'
            )
            forecast_values = forecast.predicted_mean
            if forecast_values.ndim > 1:
                forecast_values = forecast_values.values.ravel()

            forecast_df = pd.DataFrame({
                'Date': forecast_index,
                col: forecast_values
            })

            full_df = pd.concat([ts.reset_index(), forecast_df], ignore_index=True)

            ax.plot(forecast_df['Date'], forecast_df[col], label=f'Forecast')
            ax.set_title(f'({p},{d},{q}) - AIC: {aic:.1f}')
            ax.legend()

            if aic < best_aic:
                best_aic = aic
                best_order = (p, d, q)
                best_result = result
                best_forecast_df = full_df.copy()

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(f"{OUTPUT_DIR}/sarima_{label}_{col}.png", dpi=300)
        plt.close()

        if best_forecast_df is not None:
            best_forecast_df = best_forecast_df.rename(columns={col: f'{col}'})
            if df_best_all.empty:
                df_best_all = best_forecast_df
            else:
                df_best_all = pd.merge(df_best_all, best_forecast_df, on='Date', how='outer')

    df_best_all.to_csv(f"{OUTPUT_DIR}/best_sarima_{label}.csv", index=False, encoding='utf-8-sig')
    print(f"[{label}] 완료: best_sarima_{label}.csv 저장됨.")

# 실행
process_and_forecast(INPUT_FILES['seasonal'], 'seasonal')
process_and_forecast(INPUT_FILES['trend'], 'trend')
