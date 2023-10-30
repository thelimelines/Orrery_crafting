from PyQt6.QtWidgets import QApplication, QWidget, QSlider, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QTextEdit, QFrame
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QBrush
import sys
import numpy as np
import csv
from itertools import zip_longest

class CelestialBody:
    def __init__(self, parent, a, e, T, color, sidereal_rotation_period=None, radius=None, mass=None, inclination=None, lon_of_asc_node=None, arg_of_periapsis=None, albedo=None):
        self.parent = parent
        self.a = a
        self.e = e
        self.T = T
        self.color = color
        self.trail = []
        self.sidereal_rotation_period = sidereal_rotation_period
        self.radius = radius
        self.mass = mass
        self.inclination = inclination
        self.lon_of_asc_node = lon_of_asc_node
        self.arg_of_periapsis = arg_of_periapsis
        self.albedo = albedo

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

        self.bodies = []
        self.bodies_by_name = {}
        self.load_celestial_bodies_from_csv("Organised_sourced_orrery_data.csv")

        # Calculate max distance for dynamic scaling
        self.max_distance = max([body.a * (1 + body.e) for body in self.bodies])

        self.initUI()

    def load_celestial_bodies_from_csv(self, csv_file):
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Skip the header row
            for row in reader:
                body_name = row[0]  # Unpack body_name first
                try:
                    # Use zip_longest to fill in missing values with None
                    unpacked_row = list(zip_longest(headers, row, fillvalue=None))
                    
                    # Unpack and convert to appropriate types or set to None if data is missing
                    body_name, parent_name, mass, radius, a, T, inclination, e, sidereal_rotation_period, _, color = [unpacked_row[i][1] for i in range(11)]
                    a, e, T, mass, radius, inclination, sidereal_rotation_period = map(lambda x: float(x.replace("E", "e")) if x else None, [a, e, T, mass, radius, inclination, sidereal_rotation_period])
                    r, g, b = map(int, color.split('-')) if color else (None, None, None)

                    # Convert units to SI
                    if a is not None:
                        a *= 1e3  # km to m
                    if T is not None:
                        T *= 24 * 60 * 60  # days to seconds
                    if sidereal_rotation_period is not None:
                        sidereal_rotation_period *= 60 * 60  # hrs to seconds
                    
                    # Check for parent
                    parent = self.bodies_by_name.get(parent_name) if parent_name and parent_name != 'None' else None
                    
                    # Check for essential attributes for rendering
                    if None in [a, e, T, r, g, b]:
                        print(f"Skipping {body_name} due to missing essential data.")
                        continue

                    body = CelestialBody(parent, a, e, T, QColor(r, g, b), sidereal_rotation_period, radius, mass, inclination, None, None, None)
                    self.bodies.append(body)
                    self.bodies_by_name[body_name] = body
                except Exception as e:
                    print(f"Error processing {body_name}: {e}")

    def initUI(self):
        self.setWindowTitle('Orrery')
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.GlobalColor.black)
        self.setPalette(p)

        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setMinimum(1)
        self.time_slider.setMaximum(1000)
        self.time_slider.setValue(7)
        self.time_slider.valueChanged.connect(self.updateTimeStep)

        self.label = QLabel('Weeks/s: 1')
        self.label.setStyleSheet("color: white")

        self.time_step = self.time_slider.value() * 7 * 24 * 3600
        self.label.setText(f'Weeks/s: {self.time_slider.value()}')

        self.paused = False
        self.info_display = QTextEdit()
        self.info_display.setStyleSheet("background-color: black; color: white;")
        self.info_display.setReadOnly(True)
        self.info_display.setFixedWidth(200)
        self.info_display.setVisible(False)  # Initially set to hidden
        
        self.pause_button = QPushButton('Pause')
        self.pause_button.clicked.connect(self.togglePause)
        
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.time_slider)
        hlayout.addWidget(self.label)
        hlayout.addWidget(self.pause_button)

        sim_layout = QVBoxLayout()
        sim_layout.addStretch(1)
        sim_layout.addLayout(hlayout)
        
        main_layout = QHBoxLayout()
        main_layout.addLayout(sim_layout)
        main_layout.addWidget(self.info_display)
        
        self.setLayout(main_layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateTime)
        self.timer.start(8)

        self.resize(2560, 1440)
        self.show()

    def togglePause(self):
        self.paused = not self.paused
        self.pause_button.setText('Resume' if self.paused else 'Pause')

    def resizeEvent(self, event):
        self.updateTimeStep()  # Clear trails on window resize
    
    def mousePressEvent(self, event):
        click_x, click_y = event.position().x(), event.position().y()
        found = False
        for body in self.bodies:
            x, y = body.position(self.time)
            x, y = int(self.width() / 2 + x / self.scaling_factor), int(self.height() / 2 + y / self.scaling_factor)
            if abs(click_x - x) < 5 and abs(click_y - y) < 5:
                self.showBodyInfo(body)
                found = True
                break

            for trail_x, trail_y in body.trail:
                if abs(click_x - trail_x) < 5 and abs(click_y - trail_y) < 5:
                    self.showBodyInfo(body)
                    found = True
                    break

        if not found:
            self.info_display.setVisible(False)

    def updateTimeStep(self):
        value = self.time_slider.value()
        self.time_step = value * 7 * 24 * 3600
        self.label.setText(f'Weeks/s: {value}')
        
        for body in self.bodies:
            body.trail.clear()
    
    def showBodyInfo(self, body):
        info = f"Body Information:\n"
        info += f"a: {body.a}\n"
        info += f"e: {body.e}\n"
        info += f"T: {body.T}\n"
        self.info_display.setText(info)
        self.info_display.setVisible(True)

    def updateTime(self):
        if not self.paused:
            self.time += self.time_step * (8 / 1000)
            for body in self.bodies:
                x, y = body.position(self.time)
                x, y = int(self.width() / 2 + x / self.scaling_factor), int(self.height() / 2 + y / self.scaling_factor)
                body.trail.append((x, y))
                body.trail = body.trail[-500:]

        self.update()

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setPen(Qt.PenStyle.NoPen)  # This line ensures no outline is drawn
        self.scaling_factor = self.max_distance / (self.width() / 2.1)
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