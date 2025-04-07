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
ROOT_OUTPUT_DIR = 'sarima_forecast_results'

START_DATE = pd.to_datetime('2019-01-01')
TARGET_END_DATE = pd.to_datetime('2025-12-01')
SARIMA_ORDERS = [(p, d, q) for p in range(3) for d in range(3) for q in range(3)]

def fit_sarimax(ts, order, forecast_steps):
    try:
        model = SARIMAX(ts, order=order, seasonal_order=order + (12,))
        result = model.fit(disp=False)
        forecast = result.get_prediction(start=len(ts), end=len(ts) + forecast_steps)

        forecast_values = forecast.predicted_mean
        if forecast_values.ndim > 1:
            forecast_values = forecast_values.values.ravel()

        forecast_index = pd.date_range(
            start=ts.index[-1] + pd.DateOffset(months=1),
            periods=forecast_steps + 1,
            freq='MS'
        )

        forecast_df = pd.DataFrame({'Date': forecast_index, ts.name: forecast_values})
        return pd.concat([ts.reset_index(), forecast_df], ignore_index=True), ts.index[-1]
    except Exception as e:
        print(f"SARIMA 실패: {ts.name}, order={order}, error={e}")
        return None, None

def process_and_forecast(input_file, label):
    df_raw = pd.read_csv(input_file)
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    df_raw = df_raw[df_raw['Date'] >= START_DATE]

    var_columns = [col for col in df_raw.columns if col != 'Date']
    skipped_vars = []

    label_dir = os.path.join(ROOT_OUTPUT_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    for col in tqdm(var_columns, desc=f'Processing {label}'):
        df_data = df_raw[['Date', col]].dropna()
        df_data = df_data[df_data[col] != 0]
        if df_data.empty:
            skipped_vars.append(col)
            continue

        ts = df_data.set_index('Date')[col].asfreq('MS').interpolate('linear')
        if ts.index[-1] >= TARGET_END_DATE:
            skipped_vars.append(col)
            continue

        forecast_steps = ((TARGET_END_DATE.year - ts.index[-1].year) * 12 +
                          (TARGET_END_DATE.month - ts.index[-1].month))

        var_dir = os.path.join(label_dir, col)
        os.makedirs(var_dir, exist_ok=True)

        forecast_collection = pd.DataFrame({'Date': pd.date_range(start=ts.index[0], end=TARGET_END_DATE, freq='MS')})
        fig, axes = plt.subplots(9, 3, figsize=(18, 30), sharex=True)
        axes = axes.flatten()
        plt.suptitle(f'SARIMA Forecasts: {col}', fontsize=16)

        for i, (p, d, q) in enumerate(SARIMA_ORDERS):
            ax = axes[i]
            ax.plot(ts.index, ts.values, label='Observed')

            combined, split_point = fit_sarimax(ts, (p, d, q), forecast_steps)
            if combined is not None:
                ax.plot(combined['Date'], combined[col], label=f'({p},{d},{q})')

                # 예측값만 주황색으로 따로 그림
                observed = combined[combined['Date'] <= split_point]
                forecast = combined[combined['Date'] > split_point]

                # 개별 plot 저장
                plt.figure(figsize=(10, 4))
                plt.plot(observed['Date'], observed[col], label='Observed')
                plt.plot(forecast['Date'], forecast[col], label='Forecast')
                plt.title(f'{col} - SARIMA({p},{d},{q})')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                indiv_path = os.path.join(var_dir, f"{col}_{p}_{d}_{q}.png")
                plt.savefig(indiv_path, dpi=300)
                plt.close()

                # 결과 누적
                combined = combined.rename(columns={col: f'{col}_{p}_{d}_{q}'})
                forecast_collection = pd.merge(
                    forecast_collection, combined[['Date', f'{col}_{p}_{d}_{q}']],
                    on='Date', how='left'
                )

            ax.set_title(f'({p},{d},{q})')
            ax.legend()
            ax.grid(True)

        # subplot 저장
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(os.path.join(var_dir, f"sarima_{label}_{col}.png"), dpi=300)
        plt.close()

        # csv 저장
        forecast_collection.to_csv(
            os.path.join(var_dir, f"sarima_{label}_{col}.csv"),
            index=False, encoding='utf-8-sig'
        )

    # 생략 변수 기록
    if skipped_vars:
        pd.DataFrame({'Skipped Variables': skipped_vars}).to_csv(
            os.path.join(label_dir, f"skipped_variables_{label}.csv"),
            index=False, encoding='utf-8-sig'
        )
        print(f"예측 생략 변수 {len(skipped_vars)}개 기록 완료 → skipped_variables_{label}.csv")

    print(f"[{label}] 완료: 예측 결과 및 시각화 저장됨.")

# 실행
process_and_forecast(INPUT_FILES['seasonal'], 'seasonal')
process_and_forecast(INPUT_FILES['trend'], 'trend')
