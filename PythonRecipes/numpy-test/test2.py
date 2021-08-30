import traceback

from collections import defaultdict
import pprint

d = defaultdict(int)


def test():
	# L = [1, 2, 3, 4, 2, 4, 1, 2]
	#
	# for i in L:
	# 	# print(i)
	# 	d[i] += 1
	# pprint.pprint(d, width=2)
	try:
		a = [1, 2, 3, 4]
		value = a[5]
	except:
		traceback.print_exc()


if __name__ == "__main__":
	test()