# 2. Extract weights
w_trend = model_trend.linear.weight.detach().cpu().numpy().flatten()
w_seasonal = model_seasonal.linear.weight.detach().cpu().numpy().flatten()

C:\Users\SKSiltron\AppData\Local\Temp\ipykernel_16236\1070798483.py:6: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(path, map_location='cpu'))
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
Cell In[8], line 2
      1 # 2. Extract weights
----> 2 w_trend = model_trend.linear.weight.detach().cpu().numpy().flatten()
      3 w_seasonal = model_seasonal.linear.weight.detach().cpu().numpy().flatten()

File c:\Users\SKSiltron\Desktop\project\signpost_project\signpost_project_env\Lib\site-packages\torch\nn\modules\module.py:1931, in Module.__getattr__(self, name)
   1929     if name in modules:
   1930         return modules[name]
-> 1931 raise AttributeError(
   1932     f"'{type(self).__name__}' object has no attribute '{name}'"
   1933 )

AttributeError: 'DLinear_with_emb' object has no attribute 'linear'
