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

def matrix():
	conf_matrix = [[1323, 32, 3, 5, 23, 4],
	               [19, 5816, 0, 6, 0, 281],
	               [0, 0, 22306, 44, 0, 0],
	               [0, 201, 5, 1705, 2, 4],
	               [60, 8, 0, 1, 2153, 13],
	               [2, 409, 0, 0, 2, 3075]]

	conf_matrix = np.array(conf_matrix).T.tolist()
	list2 = [[1.0, 0.2270499836654688, 0.0621923937360179, 0.7250912884715701, 0.6219239373601789, 0.3985091743119266],
	         [4.4043165467625895, 1.0, 0.2739149888143177, 3.193531559728743, 2.7391498881431766, 1.7551605504587156],
	         [16.07913669064748, 3.650767722966351, 1.0, 11.6588, 41940532081, 10.0, 6.407683486238532],
	         [1.379136690647482, 0.3131329630839595, 0.08577181208053691, 1.0, 0.8577181208053691, 0.549598623853211],
	         [1.6079136690647482, 0.3650767722966351, 0.1, 1.165884194053208, 1.0, 0.6407683486238532],
	         [2.5093525179856115, 0.569748, 0.15606263982102908, 1.819509650495566, 1.5606263982102908, 1.0]]


	# result = [[sum(a * b for a, b in zip(X_row, Y_col)) for Y_col in zip(*list2)] for X_row in conf_matrix]
	# print(result)

	# output2 = [[np.dot(b, a.T) for a, b in zip(*l)] for l in zip(_confusion_matrix, list2)]

	for l in zip(conf_matrix, list2):
		print(l)
	output1 = [[a*b for a, b in zip(*l)] for l in zip(conf_matrix, list2)]
	# output1 = [for l in [a * b for a, b in zip(*l)] for l in zip(conf_matrix, list2)]
	output2 = [sum(l) for l in output1]
	print(output1)
	# output1 =  [sum([a*b]) for a, b in zip(*l)] for l in zip(conf_matrix, list2)
	print("\n======\n", output2)
	output3 = [np.diag(conf_matrix)/output2]
	print(output3)

if __name__ == '__main__':
	# array()
	matrix()