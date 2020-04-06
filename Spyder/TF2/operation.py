import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np




def my_func(a):
    return (a[0])
b = np.array([[1,2,3], [4,5,6], [7,8,9]])
  
print(np.apply_along_axis(my_func, 1, b))
   
   