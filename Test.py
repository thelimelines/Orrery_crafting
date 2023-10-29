from PyQt5.QtWidgets import QApplication, QSlider
from PyQt5.QtCore import Qt
import sys

app = QApplication(sys.argv)
slider = QSlider(Qt.Horizontal)
slider.show()
sys.exit(app.exec_())
