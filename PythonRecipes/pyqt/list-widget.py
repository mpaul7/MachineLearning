""":parameter
resource:  https://stackoverflow.com/questions/61336034/display-data-from-list-in-label-python
"""
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog, QLineEdit, QPushButton, QLabel, QGridLayout


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.setCentralWidget(self.label)

        self.resize(640, 480)

        listT = list('one', 'two', 'three', 'four')

        self.listT_iterator = iter(listT)

        self.timer = QtCore.QTimer(timeout=self.on_timeout, interval=1000)
        self.timer.start()

        self.on_timeout()

    @QtCore.pyqtSlot()
    def on_timeout(self):
        try:
            value = next(self.listT_iterator)
            self.label.setText(value)
        except StopIteration:
            self.timer.stop()

def test1():
	def init_apply_dialog(self):
		print("apply fitler")
		self.dialog = QDialog()
		self.le_filter_query = QLineEdit()
		self.bt_apply = QPushButton('Apply', self.dialog)
		# self.bt_apply.setGeometry(200, 150, 40, 20)
		self.bt_apply.show()
		self.lb_feature = QLabel("Available Features: Keyword - Range[min:max]")
		self.lb_info_features = QLabel()
		l1 = ['one', 'two', 'three', 'four']
		self.lb_info_features.setText('one')

		# bt_apply.clicked.connect(self.filter_trigger)

		self.layout_grid = QGridLayout()
		self.layout_grid.setSpacing(10)

		self.layout_grid.addWidget(self.le_filter_query, 0, 1, 1, 2)
		self.layout_grid.addWidget(self.bt_apply, 2, 1, 1, 2)
		self.layout_grid.addWidget(self.lb_feature, 3, 1, 1, 2)
		self.layout_grid.addWidget(self.lb_info_features, 4, 1, 1, 2)

		self.dialog.setLayout(self.layout_grid)
		# self.dialog.setLayout(self.layout_grid)
		# self.bt_apply.clicked.connect(lambda: self.confirm_filter(FEATURE_DISP_MAP.get(
		#     self.cb_feature.currentText()),
		#     self.cb_filter.currentText()
		# ))


		self.dialog.setWindowTitle("Apply Filter")
		self.dialog.setGeometry(800, 400, 500, 100)
		self.dialog.setWindowModality(Qt.ApplicationModal)
		self.dialog.exec_()


if __name__ == "__main__":
    # app = QtWidgets.QApplication(sys.argv)
    # w = MainWindow()
    # w.show()
    # sys.exit(app.exec_())
    test1()