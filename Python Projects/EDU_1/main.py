import numpy as np

##arrays in numpy
arr1d = np.array([-1,2,5],dtype=np.float32)
arr2d = np.array([[-1,2,5],[3,4,5]],dtype=np.float32)

#arrays are reference type
#a = arr1d
#print(arr1d)
#a[0] = 8
#print(arr1d)

##copying arrays
#d = arr2d.copy()
#print(d)

##casting arrays
#a = arr1d
#print(a.dtype)
#a = a.astype(dtype=np.int32)
#print(a.dtype)

#null but for FLOATS in arrays
#a = np.array([1,2,np.nan,5])
#print(a)

#infinity is represented like
#np.inf or -np.inf

#arr = np.array([np.nan,2,3,4,5])
#print(arr)

#creating an array by limits
#arr = np.arange(1,6,2)#from, to, step
#print(arr)

#creating an array by elements count rather than step
#arr = np.linspace(1,6,4)
#print(arr)

#np.reshape, is reshaping data
#exammple 1d with 12 elements can be turned into (4,3) or (3,4)
#but never for (4,4) or (3,3)

#arr = np.arange(8)
#reshaped = np.reshape(arr,(2,4))
#print(reshaped)
#reshaped = np.reshape(arr,(4,2))
#print(reshaped)

##note that flatten is the opposite of reshape 2D to 1D or 5D to 1D and so on
#flattened = reshaped.flatten()
#print(flattened)


#np.transpose(arr) to get the transpose

#ones and zesroes
#arr = np.zeros(4)
#arr = np.ones((2, 3))

###math on an array
# Add 1 to element values
#print(repr(arr + 1))
# Subtract element values by 1.2
#print(repr(arr - 1.2))
# Double element values
#print(repr(arr * 2))
# Halve element values
#print(repr(arr / 2))
# Integer division (half)
#print(repr(arr // 2))
# Square element values
#print(repr(arr**2))
# Square root element values
#print(repr(arr**0.5))

