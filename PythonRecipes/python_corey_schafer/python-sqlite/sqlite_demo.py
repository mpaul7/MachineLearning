import sqlite3
from employee import Employee

# for in-memory database, every time database has new copy of data. Good for testing purpose.
conn = sqlite3.connect(':memory:')
# conn = sqlite3.connect('employee.db')
c = conn.cursor()

c.execute("""CREATE TABLE employee (
			first text,
			last text,
			pay integer		
			)""")


def insert_emp(emp):
	with conn:
		c.execute("INSERT INTO employee VALUES (:first, :last, :pay)", {'first': emp_2.first, 'last':emp_2.last, 'pay':emp_2.pay})


def get_emps_by_name(lastname):
	c.execute("SELECT * FROM employee WHERE last=:last", {'last': lastname})
	return c.fetchall()


def update_pay(emp, pay):
	with conn:
		c.execute("""UPDATE employee SET pay = :pay 
					WHERE first = :first AND last = :last""",
		          {'first': emp.first, 'last': emp.last, 'pay':pay})

def remove_emp(emp):
	with conn:
		c.execute("DELETE from employee WHERE first = :first AND last = :last",
		          {'first': emp.first, 'last':emp.last})



emp_1 = Employee('Jhon', 'Doe', 80000)
emp_2 = Employee('Jane', 'Doe', 90000)

insert_emp(emp_1)
insert_emp(emp_2)

emps = get_emps_by_name('Doe')
print(emps)

update_pay(emp_2, 95000)
emps2 = get_emps_by_name('Doe')
print(emps2)
emps3 = c.execute("SELECT * from employee")
rows = emps3.fetchall()

for row in rows:
	print(row)
# c.execute("INSERT INTO employee VALUES ('{}}', '{}', 58000)") # first way, but is it not reccommended
# c.execute("INSERT INTO employee VALUES (?,?,?)", (emp_1.first, emp_1.last, emp_1.pay))
# conn.commit()
# c.execute("INfSERT INTO employee VALUES (:first, :last, :pay)", {'first': emp_2.first, 'last':emp_2.last, 'pay':emp_2.pay})
# conn.commit()
# print(emp_1.first)
# print(emp_1.last)
# print(emp_1.pay)
# add employee to the employee table
# c.execute("INSERT INTO employee VALUES ('Corey', 'Schafer', 58000)")
# c.execute("SELECT * FROM employee WHERE last='Schafer'")
# print(c.fetchone())
#
# c.execute("SELECT * FROM employee WHERE last='Doe'")
# print(c.fetchone())
# c.execute("SELECT * FROM employee WHERE last='Schafer'")
# print(c.fetchone())
# c.execute("SELECT * FROM employee WHERE last=?", ('Schafer',))
# print(c.fetchall())
# c.execute("SELECT * FROM employee WHERE last=:last", {'last':'Doe'})
# print(c.fetchall())
conn.close()