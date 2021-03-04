
from PyQt5.QtWidgets import QMessageBox

class Ui_MainWindow(object):
	...
	def show_popup(self):
		msg = QMessageBox()
		msg.setWindowTitle("Tutorial on PyQt5")
		msg.setText("This is the main text!")
		msg.setIcon(QMessageBox.Question)
		msg.setStandardButtons(QMessageBox.Cancel|QMessageBox.Retry|QMessageBox.Ignore)
		msg.setDefaultButton(QMessageBox.Retry)
		msg.setInformativeText("informative text, ya!")

		msg.setDetailedText("details")

		msg.buttonClicked.connect(self.popup_button)

	def popup_button(self, i):
		print(i.text())

def main():
	pop = Ui_MainWindow()
	pop.show_popup()


if __name__ == '__main__':
	main()