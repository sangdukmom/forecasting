import pandas as pd

# [1] 경로 설정 (실제 파일 경로로 수정)
total_data_path = "250408_total_data.csv"
seasonal_select_path = "select_feature_seasonal.csv"
trend_select_path = "select_feature_trend.csv"
output_path = "250408_selected_features_only.csv"

# [2] 전체 데이터 로딩
df_total = pd.read_csv(total_data_path, encoding='cp949')

# [3] 컬럼명에서 p_q_r 제거 (X236_5 같은 건 유지)
def extract_base_x_names(csv_path):
    df = pd.read_csv(csv_path, encoding='cp949')
    x_columns = df.columns.tolist()
    base_names = set()

    for col in x_columns:
        parts = col.split("_")
        # 뒤 3개가 모두 숫자일 경우에만 p_q_r로 간주하고 제거
        if len(parts) >= 4 and all(p.isdigit() for p in parts[-3:]):
            base_names.add("_".join(parts[:-3]))
        else:
            base_names.add(col)
    return base_names

# [4] 선택된 X인자 추출
x_seasonal = extract_base_x_names(seasonal_select_path)
x_trend = extract_base_x_names(trend_select_path)

# [5] 전체에서 해당 컬럼만 추출
selected_xs = x_seasonal.union(x_trend)
existing_columns = [col for col in df_total.columns if col in selected_xs]

# [6] 결과 저장
df_selected = df_total[existing_columns]
df_selected.to_csv(output_path, index=False, encoding='cp949')

print(f"✅ 추출 완료: {len(existing_columns)}개 컬럼 저장됨 → {output_path}")
