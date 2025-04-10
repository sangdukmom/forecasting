
import os

# 1. 텍스트파일에서 선택된 변수명 로딩
def load_selected_vars(txt_path):
    selected = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("_")
            base_name = "_".join(parts[:-3]) if len(parts) > 3 else line
            selected.append(base_name)
    return set(selected)

# 2. 실제 생성된 사리마 결과 폴더 내 변수명(폴더명) 리스트
def load_generated_vars(folder_path):
    return set([
        name for name in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, name))
    ])

# 3. 비교
txt_path = "selected_sarima_variables.txt"
seasonal_folder = "sarima_forecast_results/seasonal"
trend_folder = "sarima_forecast_results/trend"

selected_vars = load_selected_vars(txt_path)
seasonal_vars = load_generated_vars(seasonal_folder)
trend_vars = load_generated_vars(trend_folder)

missing_in_seasonal = selected_vars - seasonal_vars
missing_in_trend = selected_vars - trend_vars

print("✅ 누락된 변수 (seasonal):")
for var in sorted(missing_in_seasonal):
    print(" -", var)

print("\n✅ 누락된 변수 (trend):")
for var in sorted(missing_in_trend):
    print(" -", var)
