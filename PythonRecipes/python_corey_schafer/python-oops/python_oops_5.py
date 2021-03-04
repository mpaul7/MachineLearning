""":parameter
Special (Magic/Dunder) Methods

https://www.youtube.com/watch?v=3ohzBxoFHAY
"""


class Employee:
	raise_amount = 1.04
	num_of_emps = 0

	def __init__(self, first, last, pay):
		self.first = first
		self.last = last
		self.pay = pay
		self.email = first + '.' + last + '.company.com'
		Employee.num_of_emps += 1

	def fullname(self):
		return "{} {}".format(self.first, self.last)

	def apply_raise(self):
		self.pay = int(self.pay * self.raise_amount)

	""":parameter
	representation of the object
	Its like a toString method in Java
	"""
	def __repr__(self):
		return "Employee('{}', '{}', '{}')".format(self.first, self.last, self.pay)

	def __str__(self):
		return '{} - {}'.format(self.fullname(), self.email)

	def __add__(self, other):
		return self.pay + other.pay

	def __len__(self):
		return len(self.fullname())

emp_1 = Employee('Jhon', 'Paul', 50000)
emp_2 = Employee('Alex', 'Li', 60000)

print("==================")
print(emp_1)
print(repr(emp_1))
print(str(emp_1))

print("==================")
print(emp_1.__repr__())
print(emp_1.__str__())

print("==================")
print(emp_1 + emp_2)

print("==================")
print(len(emp_1))