---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
Cell In[7], line 2
      1 # 실행
----> 2 process_and_forecast(INPUT_FILES['seasonal'], 'seasonal')
      3 process_and_forecast(INPUT_FILES['trend'], 'trend')

Cell In[6], line 15
     12 plt.suptitle(f'SARIMA Forecast: {col}')
     14 for i, (p, d, q) in enumerate(SARIMA_ORDERS):
---> 15     ax = axes[i // 3][i % 3]
     16     ax.plot(df_data['Date'], df_data[col], label='Observed')
     18     forecast_df = forecast_sarimax(df_data, col, (p, d, q), TARGET_END_DATE)

IndexError: index 3 is out of bounds for axis 0 with size 3
