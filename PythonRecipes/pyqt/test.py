from PyQt5.QtWidgets import (QToolBar, QWidget, QMainWindow,
                             QApplication, QAction, QFileDialog,
                             QDockWidget, QListWidget, QMenu,
                             QActionGroup, QDialog, QPushButton,
                             QComboBox, QGridLayout, QLineEdit,
                             QLabel, QListWidgetItem, QMessageBox,
                             QHBoxLayout, QVBoxLayout, QFrame,
                             QDoubleSpinBox, QRadioButton, QSplitter,
                             QTableView, QTableWidget, QTableWidgetItem,
                             QAbstractItemView, QPlainTextEdit, QSpinBox, QToolButton)
from PyQt5.QtGui import QIcon, QRegExpValidator, QIntValidator, QDoubleValidator, QPixmap
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QObject

def init_exit_menu():
	dialog = QDialog()
	exit_lb = QLabel("Save Current Session")

	bt_yes = QPushButton('Yes', dialog)
	bt_yes.show()
	bt_no = QPushButton('No', dialog)
	# bt_no.clicked.connect(dialog.close)
	bt_no.show()

	# layout_grid = QGridLayout()
	layout_grid = QHBoxLayout()
	# layout_grid.setSpacing(10)
	# layout_grid.setRowStretch(15, 10)
	# layout_grid.setColumnStretch(10, 10)
	# layout_grid.addWidget(exit_lb, 0, 1, 1, 2)
	# layout_grid.addWidget(bt_yes, 1, 1, 1, 2)
	# layout_grid.addWidget(bt_no, 2, 1, 1, 2)

	layout_grid.addWidget(exit_lb)
	layout_grid.addWidget(bt_yes)
	layout_grid.addWidget(bt_no)

	dialog.setLayout(layout_grid)
	dialog.setWindowTitle("Exit Labeler")
	dialog.setWindowModality(Qt.ApplicationModal)

	dialog.exec_()


def exit_application(self):
	# choice = QMessageBox.question(self, 'Exit App',
	#                                     "Save Current Session",
	#                                     QMessageBox.Yes | QMessageBox.No)
	# print(choice)
	# if choice == QtGui.QMessageBox.Yes:
	#     self.save_prj_ui.init_dialog(self)
	#     print("Saved current session!")
	#     sys.exit()
	# elif choice == QtGui.QMessageBox.No:
	#     print("pressed no")
	#     sys.exit()

	# =================================
	msg = QMessageBox()
	msg.setIcon(QMessageBox.Question)
	msg.setInformativeText("Save Current Session")
	msg.setWindowTitle("Exit App")
	msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
	choice = msg.exec_()

	if choice == QMessageBox.Ok:
		# QMessageBox.releas
		# self.save_prj_ui.init_dialog(self)
		print(QMessageBox.objectName())
		print("Saved current session!")
	# sys.exit()
	elif choice == QMessageBox.Cancel:
		print(QMessageBox.objectName())
		print("pressed no")
	# sys.exit()


# msg.buttonClicked.connect(msgbtn)

if __name__ == '__main__':
	init_exit_menu()