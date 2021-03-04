import numpy as np
import os
def array():
	list  = [1,3,5,76,87,5,4,3,3,4,7]
	arr = np.asarray(list)
	print(arr)

	_array = array[:,1]
	print(_array)

def os_module():
	print(dir(os))
	print(os.getcwd())

if __name__ == '__main__':
	# array()
	os_module()