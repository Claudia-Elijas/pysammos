import numpy as np 

array = np.array([2,3,4], dtype=np.int64)
print(f'array: {array}, dtype: {array.dtype}')
array32 = array.astype(np.int32)

print(f'array: {array32}, dtype: {array32.dtype}')