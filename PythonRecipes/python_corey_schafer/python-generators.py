

def square_numbers1(nums):
	result = []
	for i in nums:
		result.append(i*i)
	return result


def square_numbers2(nums):
	for i in nums:
		yield (i*i)


my_nums1 = square_numbers1([1,2,3,4,5])
my_nums2 = square_numbers1([1,2,3,4,5])
""":parameter
List omprehension
"""
my_nums3 = [x*x for x in [1,2,3,4,5]]

print("================")
print(my_nums1)

print("================")
print(list(my_nums2))
for num in my_nums2:
	print(num)

print("================")
print(my_nums3)