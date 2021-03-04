""":parameter
Python filter function
Resources: https://www.youtube.com/watch?v=VrHcTFAkkak&list=PLZoTAELRMXVPBTrWtJkn3wWQxZkmTXGwe&index=19
"""


def even(num):
	if num % 2 == 0:
		return True


lst = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]

output = list(filter(even, lst))
print(output)

"""
Convert above fucntionality using lambda function
"""

output2 = list(filter(lambda num: num % 2 == 0, lst))
print(output2)
