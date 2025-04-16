import pandas as pd
import matplotlib.pyplot as plt

date = '250415'
model = 'LSTM'
target = 'y6'

trend_df = pd.read_csv('C:/Users/SKSiltron/Desktop/project/signpost_project/1_data/forecasting_results/best_results/250415_y6/trend_LSTM_12.csv')
seasonal_df = pd.read_csv('C:/Users/SKSiltron/Desktop/project/signpost_project/1_data/forecasting_results/best_results/250415_y6/seasonal_LSTM_7.csv')

trend_df.rename(columns={'pred':'trend'}, inplace=True)
seasonal_df.rename(columns={'pred':'seasonal'}, inplace=True)

df = pd.merge(seasonal_df, trend_df)

df['y6'] = df['trend'] + df['seasonal']

df['date'] = pd.to_datetime(df['date'])

# 그래프 크기 설정
plt.figure(figsize=(15, 7))

# 그래프 그리기
plt.plot(df['date'], df['y6'], label='y6', linewidth=2)
plt.plot(df['date'], df['trend'], label='trend', linestyle='--')
plt.plot(df['date'], df['seasonal'], label='seasonal', linestyle=':')

# 그래프 꾸미기
plt.title('Forecast Components over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(f'{date}_{model}_{target}.png', dpi=300) # 고해상도 저장

# 보여주기
plt.show()

df.to_csv(f'{date}_{model}_{target}.csv', index=False)
