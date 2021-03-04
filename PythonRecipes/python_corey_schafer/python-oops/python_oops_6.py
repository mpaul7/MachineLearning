""":parameter
Property Decorators - Getters, Setters, and Deleters
https://www.youtube.com/watch?v=jCzT9XFZ5bw
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

    @fullname.setter
    def fullname(self, name):
        first, last = name.split(' ')
        self.first = first
        self.last = last

    @fullname.deleter
    def fullname(self):
        print('Delete Name!')
        self.first = None
        self.last = None


emp_1 = Employee('Jhon', 'Paul', 50000)
emp_2 = Employee('Alex', 'Li', 60000)

print("==================")
emp_1.first = 'Jim'
print(emp_1.email)
print(emp_1.fullname)

print("==================")
emp_1.fullname = 'Corey Schafer'
print(emp_1.email)
print(emp_1.fullname)


print("==================")
del emp_1.fullname
print(emp_1.email)
print(emp_1.fullname)