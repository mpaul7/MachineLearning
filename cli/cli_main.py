from pathlib import Path

import click


class AppContext:
	APP_DIR = ".twc"
	MODEL_DIR = "models"
	TEMPLATE_DIR = "templates"

	def __init__(self):
		self.home_dir = Path.home() / AppContext.APP_DIR
		self.model_dir = self.home_dir / AppContext.MODEL_DIR
		self.template_dir = self.home_dir / AppContext.TEMPLATE_DIR

		self.home_dir.mkdir(exist_ok=True)
		self.model_dir.mkdir(exist_ok=True)
		self.template_dir.mkdir(exist_ok=True)

	def template(self, name):
		return (self.template_dir / name).with_suffix('.json')

	def model(self, name):
		return self.model_dir / name


# @click.group()
# @click.pass_context
def main():
	"""Traffic Wiz Classifier"""
	obj = AppContext()
	print(obj.home_dir)
	print(obj.model_dir)
	print(obj.template_dir)

if __name__ == '__main__':
	main()