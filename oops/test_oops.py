

class FeatureHelper:
	flow_Count = 0

	def __init__(self):
		FeatureHelper.flow_Count += 1

	def display_flow_count(self):
		print(f'Total flows: {FeatureHelper.flow_Count}')

class Flow:
	# obj = FeatureHelper()

	def f(self):
		print('hello world')


def flow():
	# x = Flow()
	for i in range(0, 5):
		print(i)
		FeatureHelper()
	print(FeatureHelper.flow_Count)
	obj1 = FeatureHelper()
	obj1.display_flow_count()

if __name__ == '__main__':
	flow()


class FeatureHelper:
	flow_Count = 0

	def __init__(self):
		FeatureHelper.flow_Count += 1

	def display_flow_count(self):
		print(f'Total flows: {FeatureHelper.flow_Count}')
