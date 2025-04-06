import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import os
import re
import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

dir_in = 'FEATURE_SEASON_BEV'
dir_out = 'sarima_bev_season'
# dir_out = f'{dir_in}_sarima'

data_class = 'seasonal'        # trend, seasonal, residual
data_index = 2              # 1, 2

sarima_order = [3,3,3]

def forecast_sarimax(df, column, order):
    model = SARIMAX(df[column], order=order, seasonal_order=order + (12,))
    results = model.fit(disp=False)     
    forecast = results.get_prediction(start=len(df), end=len(df) + 13)
    forecast_index = pd.date_range(start=df['date'].iloc[-1] + pd.DateOffset(months=1), periods=14, freq='MS') # 수정된 부분
    forecast_df = pd.DataFrame({column: forecast.predicted_mean, 'date': forecast_index})
    return forecast_df

columns = []
for filename in os.listdir(dir_in):
    if filename.endswith(".png"):
        match = re.match(r"(\d+\.\d{4})_(.+?)\.png", filename) 
        if match:
            columns.append(match.group(2))

# print(columns)
df_raw = pd.read_csv(f'data_{data_class}_{data_index}.csv', low_memory=False, encoding='cp949')  

df_raw['date'] = pd.to_datetime(df_raw['date'])
os.makedirs(dir_out, exist_ok = True)

y_columns = [col for col in df_raw.columns if col.startswith('Y')]
df_merged = df_raw[['date']+ y_columns]

for column in tqdm(columns):
    df_data = df_raw[['date', column]].copy()
    df_merged[column] = df_raw[column]

    fig, axes = plt.subplots(sarima_order[0], sarima_order[1], figsize=(15, 15), sharex=True)
    plt.suptitle(f'SARIMA Forecast : {column}')
    for p in range(sarima_order[0]):    
        for d in range(sarima_order[1]):
            axes[p][d].plot(df_data['date'], df_data[column], label='raw')
            for q in range(sarima_order[2]):
                order = (p, d, q)
                try:
                    forecast_df = forecast_sarimax(df_data, column, order)
                    combined_df = pd.concat([df_data, forecast_df], ignore_index=True)
                    axes[p][d].plot(combined_df['date'], combined_df[column], label=order)
                    
                    

                    combined_df = combined_df.rename(columns={column: f'{column}_{p}_{d}_{q}'})
                    df_merged = pd.merge(df_merged, combined_df, on='date', how='outer')
                except Exception as e:
                    # print(f"Error fitting SARIMA order {order}: {e}")
                    checker = 1
            axes[p][d].legend() 
            axes[p][d].grid(axis='x')   
    plt.tight_layout()
    plt.savefig(f'{dir_out}/sarima_{column}.png', dpi=300)
    for p in range(sarima_order[0]):    
        for d in range(sarima_order[1]):
            axes[p][d].set_xlim(pd.to_datetime('2022-01-01'),combined_df['date'].iloc[-1])
    plt.savefig(f'{dir_out}/sarima_{column}_detailed.png', dpi=300)
    plt.close()
                
    df_merged.to_csv(f'{dir_out}/sarima_{data_class}_{data_index}_temp.csv', index=False, encoding='cp949')
df_merged.to_csv(f'{dir_out}/sarima_{data_class}_{data_index}.csv', index=False, encoding='cp949')
