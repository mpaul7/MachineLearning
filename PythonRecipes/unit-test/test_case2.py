import unittest


class TestSum(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

    def test_sum_tuple(self):
        self.assertEqual(sum((1, 2, 2)), 5, "Should be 6")

    def test_bad_type(self):
        data = "banana"
        with self.assertRaises(TypeError):
            result = sum(data)
""":parameter
	self.assertRaises(ValueError, calc.divide, 10, 0)
	with self.assertRaises(ValueError)
		calc.divide(10,0)
"""
'''======================================================='''
# class TestBasic(unittest.TestCase):
#     def setUp(self):
#         # Load test data
#         self.app = App(database='fixtures/test_basic.json')
#
#     def test_customer_count(self):
#         self.assertEqual(len(self.app.customers), 100)
#
#     def test_existence_of_customer(self):
#         customer = self.app.get_customer(id=10)
#         self.assertEqual(customer.name, "Org XYZ")
#         self.assertEqual(customer.address, "10 Red Road, Reading")
#
#
# class TestComplexData(unittest.TestCase):
#     def setUp(self):
#         # load test data
#         self.app = App(database='fixtures/test_complex.json')
#
#     def test_customer_count(self):
#         self.assertEqual(len(self.app.customers), 10000)
#
#     def test_existence_of_customer(self):
#         customer = self.app.get_customer(id=9999)
#         self.assertEqual(customer.name, u"バナナ")
#         self.assertEqual(customer.address, "10 Red Road, Akihabara, Tokyo")

if __name__ == '__main__':
    unittest.main()