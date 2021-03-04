""":parameter
Python SQLite
Creating a Database, Table, and running Queries

https://www.youtube.com/watch?v=pd-0G0MigUA
"""


class Employee:
	raise_amount = 1.04
	num_of_emps = 0

	def __init__(self, first, last, pay):
		self.first = first
		self.last = last
		self.pay = pay

	@property
	def fullname(self):
		return "{} {}".format(self.first, self.last)

	@property
	def email(self):
		return "{}.{}@email.com".format(self.first, self.last)

	""":parameter
	representation of the object
	Its like a toString method in Java
	"""
	def __repr__(self):
		return "Employee('{}', '{}', '{}')".format(self.first, self.last, self.pay)



emp_1 = Employee('Jhon', 'Paul', 50000)
emp_2 = Employee('Alex', 'Li', 60000)

print("==================")
