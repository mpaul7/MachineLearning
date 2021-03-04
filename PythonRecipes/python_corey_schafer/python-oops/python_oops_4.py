""":parameter
Inheritance - Creating subclasses

https://www.youtube.com/watch?v=RSl87lqOXDE
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
		self.email = first + '.' + last + '@company.com'
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

class Developer(Employee):
	raise_amount = 1.10

	def __init__(self, first, last, pay, prog_lang):
		super().__init__(first,last,pay)
		self.prog_lang = prog_lang

class Manager(Employee):
	def __init__(self, first, last, pay, employees=None):
		super().__init__(first,last,pay)
		if employees is None:
			self.employees = employees
		else:
			self.employees = employees

	def add_emp(self, emp):
		if emp not in self.employees:
			self.employees.append(emp)

	def remove_emp(self, emp):
		if emp in self.employees:
			self.employees.remove(emp)
	def print_emp(self):
		for emp in self.employees:
			print('--->', emp.fullname())


dev_1 = Developer('Jhon', 'Paul', 50000, 'Java')
dev_2 = Developer('Alex', 'Li', 60000, 'Python')

mgr_1 = Manager('Sue', 'Smith', 90000, [dev_1])

print(" ==================")
print(dev_1.email)
print(dev_1.prog_lang)

print(" ==================")
print(dev_1.pay)
dev_1.apply_raise()
print(dev_1.pay)

print("===================")
print(mgr_1.email)
mgr_1.print_emp()
mgr_1.add_emp(dev_2)
mgr_1.print_emp()
mgr_1.remove_emp(dev_1)
mgr_1.print_emp()

print("===================")
print(issubclass(Developer, Employee))
print("===================")
# print(help(Developer))


