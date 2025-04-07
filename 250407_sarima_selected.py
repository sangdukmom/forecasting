import pandas as pd
import os

def load_selected_variables(txt_path, label):
    selected = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('_')
                var = '_'.join(parts[:-3])
                p, d, q = parts[-3], parts[-2], parts[-1]
                selected.append((var, f"{var}_{p}_{d}_{q}"))
    return selected

def collect_selected_series(txt_path, label, output_path):
    root = f'sarima_forecast_results/{label}'
    selected = load_selected_variables(txt_path, label)

    df_all = pd.DataFrame()

    for var_name, full_col_name in selected:
        csv_path = os.path.join(root, var_name, f"sarima_{label}_{var_name}.csv")
        if not os.path.exists(csv_path):
            print(f"[경고] 파일 없음: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if 'Date' not in df.columns or full_col_name not in df.columns:
            print(f"[경고] 필요한 컬럼 누락: {csv_path}")
            continue

        df = df[['Date', full_col_name]]
        if df_all.empty:
            df_all = df
        else:
            df_all = pd.merge(df_all, df, on='Date', how='outer')

    df_all.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"[완료] 저장됨: {output_path}")

# 실행
collect_selected_series(
    txt_path='250407_sarima_selected_pqr_seasonal.txt',
    label='seasonal',
    output_path='250407_sarima_selected_seasonal.csv'
)

collect_selected_series(
    txt_path='250407_sarima_selected_pqr_trend.txt',
    label='trend',
    output_path='250407_sarima_selected_trend.csv'
)
