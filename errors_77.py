# ====== 가중치 추출 ======
w_trend = load_fc1_fc2_weights(trend_model_path, len(trend_x_names))
w_seasonal = load_fc1_fc2_weights(seasonal_model_path, len(seasonal_x_names))

df_trend = pd.DataFrame({'X': trend_x_names, 'trend_weight': w_trend})
df_seasonal = pd.DataFrame({'X': seasonal_x_names, 'seasonal_weight': w_seasonal})
df_merged = pd.merge(df_trend, df_seasonal, on='X', how='outer').fillna(0)
df_merged['total_weight'] = df_merged['trend_weight'] + df_merged['seasonal_weight']
df_merged['abs_total_weight'] = np.abs(df_merged['total_weight'])
df_sorted = df_merged.sort_values(by='abs_total_weight', ascending=False)
df_sorted.to_csv(f"{output_dir}/dlinear_total_weight_summary.csv", index=False, encoding='cp949')

-------
C:\Users\SKSiltron\AppData\Local\Temp\ipykernel_2424\2587662126.py:6: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(path, map_location='cpu'))
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
Cell In[9], line 11
      9 df_merged['abs_total_weight'] = np.abs(df_merged['total_weight'])
     10 df_sorted = df_merged.sort_values(by='abs_total_weight', ascending=False)
---> 11 df_sorted.to_csv(f"{output_dir}/dlinear_total_weight_summary.csv", index=False, encoding='cp949')

File c:\Users\SKSiltron\Desktop\project\signpost_project\signpost_project_env\Lib\site-packages\pandas\util\_decorators.py:333, in deprecate_nonkeyword_arguments.<locals>.decorate.<locals>.wrapper(*args, **kwargs)
    327 if len(args) > num_allow_args:
    328     warnings.warn(
    329         msg.format(arguments=_format_argument_list(allow_args)),
    330         FutureWarning,
    331         stacklevel=find_stack_level(),
    332     )
--> 333 return func(*args, **kwargs)

File c:\Users\SKSiltron\Desktop\project\signpost_project\signpost_project_env\Lib\site-packages\pandas\core\generic.py:3967, in NDFrame.to_csv(self, path_or_buf, sep, na_rep, float_format, columns, header, index, index_label, mode, encoding, compression, quoting, quotechar, lineterminator, chunksize, date_format, doublequote, escapechar, decimal, errors, storage_options)
   3956 df = self if isinstance(self, ABCDataFrame) else self.to_frame()
   3958 formatter = DataFrameFormatter(
   3959     frame=df,
   3960     header=header,
   (...)
   3964     decimal=decimal,
   3965 )
-> 3967 return DataFrameRenderer(formatter).to_csv(
...
    614 parent = Path(path).parent
    615 if not parent.is_dir():
--> 616     raise OSError(rf"Cannot save file into a non-existent directory: '{parent}'")

OSError: Cannot save file into a non-existent directory: 'C:\Users\SKSiltron\Desktop\project\signpost_project\1_data\forecasting_results\best_results\250415_y6\250415_y6'
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
