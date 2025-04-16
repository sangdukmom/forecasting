# 1. Load both models
model_trend = load_model(trend_model_path, len(trend_x_names))
model_seasonal = load_model(seasonal_model_path, len(seasonal_x_names))

--------
C:\Users\SKSiltron\AppData\Local\Temp\ipykernel_1936\1070798483.py:6: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(path, map_location='cpu'))
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
Cell In[6], line 2
      1 # 1. Load both models
----> 2 model_trend = load_model(trend_model_path, len(trend_x_names))
      3 model_seasonal = load_model(seasonal_model_path, len(seasonal_x_names))

Cell In[5], line 6
      4 model_cfg['input_size'] = input_size
      5 model = Model(**model_cfg)
----> 6 model.load_state_dict(torch.load(path, map_location='cpu'))
      7 model.eval()
      8 return model

File c:\Users\SKSiltron\Desktop\project\signpost_project\signpost_project_env\Lib\site-packages\torch\nn\modules\module.py:2584, in Module.load_state_dict(self, state_dict, strict, assign)
   2576         error_msgs.insert(
   2577             0,
   2578             "Missing key(s) in state_dict: {}. ".format(
   2579                 ", ".join(f'"{k}"' for k in missing_keys)
   2580             ),
   2581         )
   2583 if len(error_msgs) > 0:
-> 2584     raise RuntimeError(
   2585         "Error(s) in loading state_dict for {}:\n\t{}".format(
   2586             self.__class__.__name__, "\n\t".join(error_msgs)
...

RuntimeError: Error(s) in loading state_dict for DLinear_with_emb:
	size mismatch for fc1.weight: copying a param with shape torch.Size([1, 247]) from checkpoint, the shape in current model is torch.Size([1, 275]).
	size mismatch for fc2.weight: copying a param with shape torch.Size([1, 247]) from checkpoint, the shape in current model is torch.Size([1, 275]).
	size mismatch for periodic.periodic.wegiht: copying a param with shape torch.Size([247, 24]) from checkpoint, the shape in current model is torch.Size([275, 24]).
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
