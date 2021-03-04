
""":parameter
Manually adding the objects for each employee.
https://www.youtube.com/watch?v=ZDa-Z5JzLYM
"""
class Employee:
	pass

emp_1 = Employee()
emp_2 = Employee()

print(emp_1)
print(emp_2)

emp_1.first = 'Jhon'
emp_1.last = 'paul'
emp_1.email = 'jhon.Paul@company.com'
emp_1.pay = 50000


emp_2.first = 'Shawn'
emp_2.last = 'Li'
emp_2.email = 'Shawn.Li@company.com'
emp_2.pay = 60000

print(emp_1.email)
print(emp_2.email)