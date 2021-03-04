""":parameter
Resource: https://www.youtube.com/watch?v=UXV_o2W_UUA&list=PLZoTAELRMXVPBTrWtJkn3wWQxZkmTXGwe&index=26
"""

class Error(Exception):
	pass

class dobException(Error):
	pass

year = 2001
age = 2021 - year
try:
	if age <= 30 & age > 20:
		print("valid age")
	else:
		raise dobException
except dobException:
	print("The year range is not valid")