from PyQt6.QtWidgets import QApplication, QWidget, QSlider, QVBoxLayout, QLabel, QHBoxLayout, QSizePolicy
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QBrush
import sys
import numpy as np

class CelestialBody:
    def __init__(self, parent, a, e, T, color):
        self.parent = parent
        self.a = a
        self.e = e
        self.T = T
        self.color = color
        self.trail = []  # To store the trail

    def position(self, time):
        theta = 2 * np.pi * (time % self.T) / self.T
        r = self.a * (1 - self.e ** 2) / (1 + self.e * np.cos(theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        if self.parent:
            px, py = self.parent.position(time)
            x += px
            y += py

        return x, y

class Orrery(QWidget):
    def __init__(self):
        super().__init__()
        self.time = 0
        self.scaling_factor = 1

        # Initialize celestial bodies
        self.sun = CelestialBody(None, 0, 0, 1, QColor(255, 255, 0))
        self.mercury = CelestialBody(self.sun, 5.791e10, 0.2056, 88.0 * 24 * 3600, QColor(169, 169, 169))
        self.venus = CelestialBody(self.sun, 1.082e11, 0.0068, 225 * 24 * 3600, QColor(255, 215, 0))
        self.earth = CelestialBody(self.sun, 1.496e11, 0.0167, 365.25 * 24 * 3600, QColor(0, 0, 255))
        self.moon = CelestialBody(self.earth, 3.844e9, 0.0549, 27.3 * 24 * 3600, QColor(128, 128, 128))
        self.mars = CelestialBody(self.sun, 2.279e11, 0.0934, 687 * 24 * 3600, QColor(255, 0, 0))

        self.bodies = [self.sun, self.mercury, self.venus, self.earth, self.moon, self.mars]


        # Calculate max distance for dynamic scaling
        self.max_distance = max([body.a * (1 + body.e) for body in self.bodies])
        
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Orrery')
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.GlobalColor.black)
        self.setPalette(p)

        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setMinimum(1)
        self.time_slider.setMaximum(100)
        self.time_slider.setValue(7)
        self.time_slider.valueChanged.connect(self.updateTimeStep)

        self.label = QLabel('Weeks/s: 1')
        self.label.setStyleSheet("color: white")

        self.time_step = self.time_slider.value() * 7 * 24 * 3600
        self.label.setText(f'Weeks/s: {self.time_slider.value()}')

        hlayout = QHBoxLayout()
        hlayout.addWidget(self.time_slider)
        hlayout.addWidget(self.label)

        layout = QVBoxLayout()
        layout.addStretch(1)
        layout.addLayout(hlayout)

        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateTime)
        self.timer.start(8)

        self.resize(800, 800)
        self.show()

    def updateTimeStep(self):
        value = self.time_slider.value()
        self.time_step = value * 7 * 24 * 3600
        self.label.setText(f'Weeks/s: {value}')
        
        for body in self.bodies:
            body.trail.clear()

    def updateTime(self):
        self.time += self.time_step * (8 / 1000)
        for body in self.bodies:
            x, y = body.position(self.time)
            x, y = int(self.width() / 2 + x / self.scaling_factor), int(self.height() / 2 + y / self.scaling_factor)
            body.trail.append((x, y))
            body.trail = body.trail[-500:]

        self.update()

    def paintEvent(self, event):
        qp = QPainter(self)
        self.scaling_factor = self.max_distance / (self.width() / 3)
        for body in self.bodies:
            qp.setBrush(QBrush(body.color, Qt.BrushStyle.SolidPattern))
            x, y = body.position(self.time)
            x, y = int(self.width() / 2 + x / self.scaling_factor), int(self.height() / 2 + y / self.scaling_factor)
            qp.drawEllipse(x, y, 5, 5)

            for i, (trail_x, trail_y) in enumerate(body.trail):
                alpha = int(255 * (i / 500))
                if alpha > 0:
                    qp.setBrush(QBrush(QColor(body.color.red(), body.color.green(), body.color.blue(), alpha), Qt.BrushStyle.SolidPattern))
                    qp.drawEllipse(trail_x, trail_y, 4, 4)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Orrery()
    sys.exit(app.exec())