---------------------------------------------------------------------------
MemoryError                               Traceback (most recent call last)
Cell In[7], line 2
      1 # 실행
----> 2 process_and_forecast(INPUT_FILES['seasonal'], 'seasonal')
      3 process_and_forecast(INPUT_FILES['trend'], 'trend')

Cell In[6], line 26
     23     ax.plot(combined['Date'], combined[col], label=f'({p},{d},{q})')
     25     combined = combined.rename(columns={col: f'{col}_{p}_{d}_{q}'})
---> 26     df_merged = pd.merge(
     27         df_merged, combined[['Date', f'{col}_{p}_{d}_{q}']],
     28         on='Date', how='outer'
     29     )
     31 ax.legend()
     32 ax.grid(True)

File c:\Users\SKSiltron\Desktop\project\signpost_project\signpost_project_env\Lib\site-packages\pandas\core\reshape\merge.py:184, in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    169 else:
    170     op = _MergeOperation(
    171         left_df,
    172         right_df,
   (...)
    182         validate=validate,
    183     )
...
   1802 return lidx, ridx

File join.pyx:189, in pandas._libs.join.full_outer_join()

MemoryError: Unable to allocate 91.1 GiB for an array with shape (12230591226,) and data type int64
