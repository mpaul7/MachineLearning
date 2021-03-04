""":parameter
Resource: https://www.youtube.com/watch?v=j1lTvjmOJbQ&list=PLZoTAELRMXVPBTrWtJkn3wWQxZkmTXGwe&index=18
"""


def even_or_odd(num):
	if num % 2 == 0:
		return True
	else:
		return False


lst = [1, 2, 3, 4, 5, 6, 7, 8]
output = list(map(even_or_odd, lst)) # lazy activation
print(output)
