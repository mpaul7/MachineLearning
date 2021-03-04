""":parameter
List comprehension: it provides a concise way to create lists
Resource: https://www.youtube.com/watch?v=iiDDZtzvMC8&list=PLZoTAELRMXVPBTrWtJkn3wWQxZkmTXGwe&index=20
"""

lst1 = []


def lst_square(lst):
	for i in lst:
		lst1.append(i * i)
	return lst1


lst = [1, 2, 3, 4, 5]
output = lst_square(lst)
print(output)

""" 
apply list copmprehension to above function
"""

output2 = [i * i for i in lst]
print(output2)

output3 = [i * i for i in lst if i % 2 !=0]
print(output3)