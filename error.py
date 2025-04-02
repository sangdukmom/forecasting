# 연도 및 분기 추출
df_x[['Year', 'Quarter']] = df_x['분기'].str.extract(r'(\d{4})\s*Q[1-4]')
df_x['Year'] = df_x['Year'].astype(int)
df_x['Quarter'] = 'Q' + df_x['Quarter']

# error

---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[8], line 2
      1 # 연도 및 분기 추출
----> 2 df_x[['Year', 'Quarter']] = df_x['분기'].str.extract(r'(\d{4})\s*Q[1-4]')
      3 df_x['Year'] = df_x['Year'].astype(int)
      4 df_x['Quarter'] = 'Q' + df_x['Quarter']

File c:\Users\SKSiltron\Desktop\project\signpost_project\signpost_project_env\Lib\site-packages\pandas\core\frame.py:4299, in DataFrame.__setitem__(self, key, value)
   4297     self._setitem_frame(key, value)
   4298 elif isinstance(key, (Series, np.ndarray, list, Index)):
-> 4299     self._setitem_array(key, value)
   4300 elif isinstance(value, DataFrame):
   4301     self._set_item_frame_value(key, value)

File c:\Users\SKSiltron\Desktop\project\signpost_project\signpost_project_env\Lib\site-packages\pandas\core\frame.py:4341, in DataFrame._setitem_array(self, key, value)
   4336 else:
   4337     # Note: unlike self.iloc[:, indexer] = value, this will
   4338     #  never try to overwrite values inplace
   4340     if isinstance(value, DataFrame):
-> 4341         check_key_length(self.columns, key, value)
   4342         for k1, k2 in zip(key, value.columns):
   4343             self[k1] = value[k2]

File c:\Users\SKSiltron\Desktop\project\signpost_project\signpost_project_env\Lib\site-packages\pandas\core\indexers\utils.py:390, in check_key_length(columns, key, value)
...
    391 else:
    392     # Missing keys in columns are represented as -1
    393     if len(columns.get_indexer_non_unique(key)[0]) != len(value.columns):

ValueError: Columns must be same length as key
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
