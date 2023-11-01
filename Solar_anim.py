from PyQt6.QtWidgets import QApplication, QWidget, QSlider, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QTextEdit, QFrame
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QBrush
import sys
from scipy.optimize import root_scalar
import numpy as np
import csv
from itertools import zip_longest

def format_number(num):
    if num is None:
        return "N/A"
    elif num == 0:
        return "0"
    else:
        order_of_magnitude = np.floor(np.log10(np.abs(num)))
        if abs(order_of_magnitude) >= 4:  # Using scientific notation for numbers >= 10,000 or <= 0.0001
            return "{:.3e}".format(num)
        else:
            # Formatting to 3 significant figures
            return "{:,.{precision}g}".format(num, precision=3)

class CelestialBody:
    def __init__(self, body_name, parent, a, e, T, color, sidereal_rotation_period=None, radius=None, mass=None, inclination=None, axial_tilt=None):
        self.body_name = body_name
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
        self.axial_tilt = axial_tilt

    def position(self, time):
        # Mean Anomaly
        M = 2 * np.pi / self.T * time

        # Solve for Eccentric Anomaly (using Newton's method)
        E = root_scalar(lambda E: E - self.e * np.sin(E) - M, x0=M, x1=M + 0.1).root

        # True Anomaly
        theta = 2 * np.arctan(np.sqrt((1 + self.e) / (1 - self.e)) * np.tan(E / 2))

        # Position
        x = self.a * (np.cos(theta) - self.e)
        y = self.a * np.sqrt(1 - self.e ** 2) * np.sin(theta)

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
                    body_name, parent_name, mass, radius, a, T, inclination, e, sidereal_rotation_period, axial_tilt, color = [unpacked_row[i][1] for i in range(11)]
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

                    body = CelestialBody(body_name,parent, a, e, T, QColor(r, g, b), sidereal_rotation_period, radius, mass, inclination)
                    # Check if grandparent is the "Sun" and modify 'a' accordingly
                    if parent and parent.parent and parent.parent.body_name == "Sun":
                        body.a *= 50
                        if parent.body_name == "Mars":
                            body.a *= 30
                        if parent.body_name == "Jupiter" or parent.body_name == "Saturn":
                            body.a *= 2
                        if parent.body_name == "Uranus" or parent.body_name == "Neptune":
                            body.a *= 10
                        if parent.body_name == "Plutonian barycentre" or parent.body_name == "Eris":
                            body.a *= 100
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
        self.info_display.setFixedWidth(250)
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

        self.resize(1440, 1440)
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
        info = f"Body Information: {body.body_name}\n\n"
        info += "--- Planetary Properties ---\n"
        info += f"Mass: {format_number(body.mass)} kg\n" if body.mass else ""
        info += f"Volumetric mean radius: {format_number(body.radius*1000)} m\n" if body.radius else ""
        info += f"Axial Tilt: {format_number(body.axial_tilt)} degrees\n" if body.axial_tilt else ""
        info += "\n--- Orbital Parameters ---\n"
        info += f"Semi-major Axis (a): {format_number(body.a)} m\n"
        info += f"Eccentricity (e): {format_number(body.e)}\n"
        info += f"Siderial orbital Period (T): {format_number(body.T)} s\n"
        info += f"Inclination: {format_number(body.inclination)} degrees\n" if body.inclination else ""
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
        self.scaling_factor = self.max_distance / (self.width() / 0.175)
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