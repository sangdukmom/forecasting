---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
File c:\Users\SKSiltron\Desktop\project\signpost_project\signpost_project_env\Lib\site-packages\pandas\core\indexes\base.py:565, in Index.__new__(cls, data, dtype, copy, name, tupleize_cols)
    564 try:
--> 565     arr = sanitize_array(data, None, dtype=dtype, copy=copy)
    566 except ValueError as err:

File c:\Users\SKSiltron\Desktop\project\signpost_project\signpost_project_env\Lib\site-packages\pandas\core\construction.py:659, in sanitize_array(data, index, dtype, copy, allow_2d)
    657             subarr = maybe_infer_to_datetimelike(subarr)
--> 659 subarr = _sanitize_ndim(subarr, data, dtype, index, allow_2d=allow_2d)
    661 if isinstance(subarr, np.ndarray):
    662     # at this point we should have dtype be None or subarr.dtype == dtype

File c:\Users\SKSiltron\Desktop\project\signpost_project\signpost_project_env\Lib\site-packages\pandas\core\construction.py:718, in _sanitize_ndim(result, data, dtype, index, allow_2d)
    717         return result
--> 718     raise ValueError(
    719         f"Data must be 1-dimensional, got ndarray of shape {data.shape} instead"
    720     )
    721 if is_object_dtype(dtype) and isinstance(dtype, ExtensionDtype):
    722     # i.e. NumpyEADtype("O")

ValueError: Data must be 1-dimensional, got ndarray of shape (420, 4) instead

The above exception was the direct cause of the following exception:
...
--> 570         raise ValueError("Index data must be 1-dimensional") from err
    571     raise
    572 arr = ensure_wrapped_if_datetimelike(arr)

ValueError: Index data must be 1-dimensional
