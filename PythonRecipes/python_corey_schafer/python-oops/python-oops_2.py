
""":parameter
Creating init method in class to automatically create objects for each employee.
https://www.youtube.com/watch?v=ZDa-Z5JzLYM
"""
class Employee:

	def __init__(self, first, last, pay):
		self.first = first
		self.last = last
		self.pay = pay
		self.email = first + '.' + last +'.company.com'

	def fullname(self):
		return "{} {}".format(self.first, self.last)


emp_1 = Employee('Jhon', 'Paul', 50000)
emp_2 = Employee('Alex', 'Li', 60000)

print(emp_1.email)
print(emp_2.email)
print(emp_1.fullname()) # when a method is callad with an instance then there is no need to pass object.
Employee.fullname(emp_1) # when a method is called with Class reference then there is a need to pass object. 