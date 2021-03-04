import sys
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

class Widget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        self.button_layout = QHBoxLayout()
        self.widget_layout = QVBoxLayout()

        for button_number in range(1, 3):
            button = QToolButton()
            button.setText(str(button_number))
            button.setObjectName('Button%d' % button_number)
            button.released.connect(self.button_released)
            self.button_layout.addWidget(button)

        self.status_label = QLabel('No button clicked')

        self.widget_layout.addItem(self.button_layout)
        self.widget_layout.addWidget(self.status_label)
        self.setLayout(self.widget_layout)

    def button_released(self):
        sending_button = self.sender()
        self.status_label.setText('%s Clicked!' % str(sending_button.objectName()))




if __name__ == '__main__':
  app = QApplication(sys.argv)

  widget = Widget()
  widget.show()

  sys.exit(app.exec_())
Share