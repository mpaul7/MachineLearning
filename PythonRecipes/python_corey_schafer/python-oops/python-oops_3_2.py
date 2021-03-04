""":parameter
class methods and static methods
regular methods automatically takes instance as the first argument, as 'self'.
How to change a method to take class as the first argument
It is done by using @classmethod decorator.

static methods do noit take instance or class as first argument
https://www.youtube.com/watch?v=rq8cL2XMM5M
"""

import datetime
import datetime

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

	@classmethod
	def set_raise_amt(cls, amount):
		cls.raise_amount = amount

	@classmethod
	def from_string(cls, emp_str):
		first, last, pay = emp_str.split('-')
		return cls(first, last, pay)

	@staticmethod
	def is_workday(day):
		if day.weekday == 5 or day.weekday() == 6:
			return False
		return True

emp_1 = Employee('Jhon', 'Paul', 50000)
emp_2 = Employee('Alex', 'Li', 60000)

print(" ==================")
print(Employee.raise_amount)
print(emp_1.raise_amount)
print(emp_2.raise_amount)

print(" ==================")
Employee.set_raise_amt(1.05)
print(Employee.raise_amount)
print(emp_1.raise_amount)
print(emp_2.raise_amount)

emp_str_1 = 'jhon-Doe-70000'
emp_str_2 = 'Steve-Smith-30000'
emp_str_3 = 'Jane-Doe-90000'

new_emp_1 = Employee.from_string(emp_str_1)

print(" ==================")
print(new_emp_1.email)
print(new_emp_1.pay)

print("===================")
my_date = datetime.date(2016, 7, 10)
print(Employee.is_workday(my_date))

