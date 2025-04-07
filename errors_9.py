---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
Cell In[8], line 3
      1 # 데이터 로딩
      2 meta_df = pd.read_csv(trend_csv_path, encoding='cp949')
----> 3 meta_df.set_index('Date', inplace=True)
      4 meta_df.index = pd.to_datetime(meta_df.index)
      5 meta_df = meta_df.interpolate(method='linear', limit_area='inside').bfill().ffill()

File c:\Users\SKSiltron\Desktop\project\signpost_project\signpost_project_env\Lib\site-packages\pandas\core\frame.py:6122, in DataFrame.set_index(self, keys, drop, append, inplace, verify_integrity)
   6119                 missing.append(col)
   6121 if missing:
-> 6122     raise KeyError(f"None of {missing} are in the columns")
   6124 if inplace:
   6125     frame = self

KeyError: "None of ['Date'] are in the columns"
