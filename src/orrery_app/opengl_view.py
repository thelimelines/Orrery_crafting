from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_FALSE,
    GL_LEQUAL,
    GL_LINE_SMOOTH,
    GL_LINE_LOOP,
    GL_LINE_STRIP,
    GL_LINES,
    GL_MODELVIEW,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_PROJECTION,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_UNPACK_ALIGNMENT,
    GL_TRIANGLE_STRIP,
    GL_UNSIGNED_BYTE,
    GL_VERTEX_ARRAY,
    glBegin,
    glBindTexture,
    glBlendFunc,
    glClear,
    glClearColor,
    glColor3f,
    glColor4f,
    glDepthFunc,
    glDepthMask,
    glDisable,
    glDisableClientState,
    glEnable,
    glEnableClientState,
    glEnd,
    glGenerateMipmap,
    glDrawArrays,
    glGenTextures,
    glGetIntegerv,
    glLineWidth,
    glLoadMatrixf,
    glMatrixMode,
    glMultMatrixf,
    glPopMatrix,
    glPushMatrix,
    glRotatef,
    glPixelStorei,
    glTranslatef,
    glTexImage2D,
    glTexParameteri,
    glVertex3f,
    glVertexPointer,
    glViewport,
    GL_FLOAT,
    GL_LINEAR,
    GL_LINEAR_MIPMAP_LINEAR,
    GL_MAX_TEXTURE_SIZE,
    GL_REPEAT,
    GL_RGBA,
    GL_TRUE,
)
from OpenGL.GLU import gluCylinder, gluDisk, gluNewQuadric, gluQuadricTexture, gluSphere
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from .gear_models import ArmRenderSpec, GearRenderSpec
from .models import BODY_BY_NAME, BodySettings, BodyState
from .textures import resolve_texture_paths


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm == 0:
        forward_norm = 1.0
    forward = forward / forward_norm
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1.0e-6:
        alt_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(np.dot(forward, alt_up)) > 0.98:
            alt_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right = np.cross(forward, alt_up)
        right_norm = np.linalg.norm(right)
    if right_norm == 0:
        right_norm = 1.0
    right = right / right_norm
    true_up = np.cross(right, forward)

    view = np.identity(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = -forward
    view[0, 3] = -np.dot(right, eye)
    view[1, 3] = -np.dot(true_up, eye)
    view[2, 3] = np.dot(forward, eye)
    return view


def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2.0 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj


class OrreryGLWidget(QOpenGLWidget):
    bodyDoubleClicked = Signal(str)
    gearClicked = Signal(str)
    glInitialized = Signal()
    texturesLoaded = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self._positions: Dict[str, BodyState] = {}
        self._settings: Dict[str, BodySettings] = {}
        self._screen_positions: Dict[str, Tuple[float, float, float]] = {}
        self._body_rotations: Dict[str, np.ndarray] = {}
        self._body_spin_dirs: Dict[str, float] = {}

        self._yaw = 0.0
        self._pitch = 60.0
        self._distance = 240.0
        self._target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._last_pos = QPoint()

        self.global_distance_scale = 1.0e-7
        self.global_radius_scale = 1.0e-4
        self.auto_distance_scale = True
        self.target_scene_radius = 140.0
        self.min_visible_radius = 0.2
        self.highlight_threshold_earth_units = 0.5
        self.size_mode = "log"  # "linear" or "log"
        self.size_log_base = 10.0
        self.earth_diameter_mm = 10.0
        self.min_body_diameter_mm = 1.0
        self.reference_radius_km = 6371.0  # Earth radius
        self.distance_mode = "mechanical"  # "mechanical" or "astronomical"
        self.show_system_bounds = False
        self.show_orbit_bands = True
        self.align_to_ecliptic = True
        self.ecliptic_tilt_deg = 23.43928
        self._zoom_min = 10.0
        self._zoom_max = 2000.0
        self._orbit_normal: Optional[np.ndarray] = None
        self._positions_scaled = False

        self._max_distance_km = 0.0
        self._quadric = None
        self._scaled_positions: Dict[str, np.ndarray] = {}
        self._system_bounds: Dict[str, float] = {}
        self._orbit_sweeps: Dict[str, Tuple[float, float, float]] = {}
        self._moon_orbit_bands: Dict[str, Tuple[Tuple[float, float], ...]] = {}
        self._moon_orbit_center: Optional[str] = None
        self.show_moon_bands = False
        self.orbit_band_opacity = 0.035
        self.show_orbit_paths = False
        self._orbit_paths: Dict[str, np.ndarray] = {}
        self._orbit_paths_scaled = False
        self._orbit_path_centers: Dict[str, Optional[str]] = {}
        self._texture_ids: Dict[str, int] = {}
        self._texture_paths = {}
        self._textures_loaded = False
        self.real_scale_mode = False
        self._gears: Dict[str, GearRenderSpec] = {}
        self._gear_screen_positions: Dict[str, Tuple[float, float, float]] = {}
        self._gear_screen_radii: Dict[str, float] = {}
        self._gear_angles: Dict[str, float] = {}
        self._mesh_links: Tuple[Tuple[np.ndarray, np.ndarray], ...] = ()
        self._arms: Dict[str, ArmRenderSpec] = {}
        self._mouse_press_pos = QPoint()
        self._dragging = False

    def set_body_settings(self, settings: Dict[str, BodySettings]) -> None:
        self._settings = settings
        self.update()

    def set_body_rotations(self, rotations: Dict[str, np.ndarray]) -> None:
        self._body_rotations = rotations
        self.update()

    def set_body_spin_directions(self, spin_dirs: Dict[str, float]) -> None:
        self._body_spin_dirs = spin_dirs
        self.update()

    def update_positions(self, positions: Dict[str, BodyState], already_scaled: bool = False) -> None:
        self._positions = positions
        self._positions_scaled = already_scaled
        self._max_distance_km = self._compute_max_distance()
        self.update()

    def set_system_bounds(
        self,
        system_bounds: Dict[str, float],
        orbit_sweeps: Dict[str, Tuple[float, float, float]],
    ) -> None:
        self._system_bounds = system_bounds
        self._orbit_sweeps = orbit_sweeps
        self.update()

    def set_moon_bands(
        self,
        center_name: Optional[str],
        bands: Dict[str, Tuple[Tuple[float, float], ...]],
    ) -> None:
        self._moon_orbit_center = center_name
        self._moon_orbit_bands = bands
        self.update()

    def set_orbit_paths(
        self,
        paths: Dict[str, np.ndarray],
        *,
        centers: Optional[Dict[str, Optional[str]]] = None,
        already_scaled: bool = False,
    ) -> None:
        self._orbit_paths = paths
        self._orbit_paths_scaled = already_scaled
        self._orbit_path_centers = centers or {}
        self.update()

    def set_orbital_plane_normal(self, normal: Optional[Tuple[float, float, float]]) -> None:
        if normal is None:
            self._orbit_normal = None
        else:
            vec = np.array(normal, dtype=np.float32)
            norm = np.linalg.norm(vec)
            self._orbit_normal = vec / norm if norm > 0 else None
        self.update()

    def orbital_plane_rotation_matrix(self) -> np.ndarray:
        return self._orbital_plane_rotation()

    def set_gears(self, gears: Dict[str, GearRenderSpec]) -> None:
        self._gears = gears
        self.update()

    def set_arms(self, arms: Dict[str, ArmRenderSpec]) -> None:
        self._arms = arms
        self.update()

    def set_gear_angles(self, angles: Dict[str, float]) -> None:
        self._gear_angles = angles
        self.update()

    def set_mesh_links(self, links: Tuple[Tuple[np.ndarray, np.ndarray], ...]) -> None:
        self._mesh_links = links
        self.update()

    def focus_on_body(self, name: str) -> None:
        state = self._positions.get(name)
        if state:
            settings = self._settings.get(name, BodySettings())
            if self._positions_scaled:
                target = np.array(state.position_km, dtype=np.float32)
            else:
                target = np.array(state.position_km, dtype=np.float32)
                if self.distance_mode != "mechanical":
                    target = target * self._current_distance_scale()
                target = target * settings.distance_scale
            if self.align_to_ecliptic:
                target = self._orbital_plane_rotation() @ target
            self._target = target
        else:
            self._target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.update()

    def initializeGL(self) -> None:
        glClearColor(0.04, 0.05, 0.08, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self._quadric = gluNewQuadric()
        gluQuadricTexture(self._quadric, True)
        self.glInitialized.emit()

    def load_textures(self) -> None:
        if self._textures_loaded:
            return
        self.makeCurrent()
        try:
            self._load_textures()
            self._textures_loaded = True
            self.texturesLoaded.emit()
        finally:
            self.doneCurrent()

    def resizeGL(self, width: int, height: int) -> None:
        glViewport(0, 0, width, height)

    def _load_textures(self) -> None:
        self._texture_paths = resolve_texture_paths()
        self._texture_ids = {}
        for name, path in self._texture_paths.items():
            try:
                tex_id = self._create_texture(path)
            except Exception as exc:  # noqa: BLE001
                continue
            if tex_id is not None:
                self._texture_ids[name] = tex_id
        if self._texture_ids:
            self.update()

    def _create_texture(self, path) -> Optional[int]:
        image = QImage(str(path))
        if image.isNull():
            return None

        image = image.convertToFormat(QImage.Format.Format_RGBA8888)
        max_size = glGetIntegerv(GL_MAX_TEXTURE_SIZE)
        try:
            max_size = int(max_size)
        except (TypeError, ValueError):
            max_size = int(max_size[0]) if max_size is not None and len(max_size) else 0
        if max_size:
            width = image.width()
            height = image.height()
            if width > max_size or height > max_size:
                scale = min(max_size / width, max_size / height)
                new_width = max(1, int(width * scale))
                new_height = max(1, int(height * scale))
                image = image.scaled(
                    new_width,
                    new_height,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )

        image = image.mirrored()
        buf = image.bits()
        try:
            buf.setsize(image.sizeInBytes())
        except AttributeError:
            pass
        try:
            data = buf.tobytes()
        except AttributeError:
            data = bytes(buf)

        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        prev_alignment = glGetIntegerv(GL_UNPACK_ALIGNMENT)
        try:
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA,
                image.width(),
                image.height(),
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                data,
            )
            glGenerateMipmap(GL_TEXTURE_2D)
        finally:
            try:
                glPixelStorei(GL_UNPACK_ALIGNMENT, int(prev_alignment))
            except (TypeError, ValueError):
                glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
        glBindTexture(GL_TEXTURE_2D, 0)
        return tex_id

    def paintGL(self) -> None:
        if self.width() <= 0 or self.height() <= 0:
            return

        painter = QPainter(self)
        if not painter.isActive():
            painter.begin(self)
        painter.beginNativePainting()

        glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))
        # QPainter can alter GL state between frames; re-assert depth testing.
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LEQUAL)

        fov_deg = 45.0
        proj = None
        eye = self._camera_eye()
        view = _look_at(eye, self._target, np.array([0.0, 1.0, 0.0], dtype=np.float32))

        self._screen_positions = {}
        self._gear_screen_positions = {}
        self._gear_screen_radii = {}
        self._scaled_positions = {}
        distance_scale = self._current_distance_scale()
        plane_rot = self._orbital_plane_rotation()

        body_radii = []
        gear_radii = []
        for name, state in self._positions.items():
            info = BODY_BY_NAME.get(name)
            if not info:
                continue
            settings = self._settings.get(name, BodySettings())
            if self._positions_scaled:
                scaled_pos = np.array(state.position_km, dtype=np.float32)
            else:
                scaled_pos = (
                    np.array(state.position_km, dtype=np.float32)
                    * distance_scale
                    * settings.distance_scale
                )
            if self.align_to_ecliptic:
                scaled_pos = plane_rot @ scaled_pos
            self._scaled_positions[name] = scaled_pos
            radius = self._scaled_radius(info.radius_km, settings.radius_scale)
            body_radii.append(radius)

        cam_dist = float(np.linalg.norm(eye - self._target))
        for gear in self._gears.values():
            gear_radii.append(gear.radius_mm)
        near, far = self._compute_clip_planes(
            body_radii + gear_radii, fov_deg, cam_dist, distance_scale
        )
        proj = _perspective(fov_deg, self.width() / self.height(), near, far)

        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(proj.T)
        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(view.T)

        # Draw bodies without blending so they stay fully opaque.
        glDisable(GL_BLEND)
        for name, state in self._positions.items():
            info = BODY_BY_NAME.get(name)
            if not info:
                continue
            settings = self._settings.get(name, BodySettings())
            scaled_pos = self._scaled_positions.get(name)
            if scaled_pos is None:
                continue
            radius = self._scaled_radius(info.radius_km, settings.radius_scale)
            draw_radius = max(radius, self.min_visible_radius)

            center_screen = self._project_point(scaled_pos, view, proj)
            self._screen_positions[name] = center_screen
            screen_radius = self._screen_radius(scaled_pos, draw_radius, view, proj, center_screen)
            show_guides = screen_radius >= 12.0

            glPushMatrix()
            glTranslatef(float(scaled_pos[0]), float(scaled_pos[1]), float(scaled_pos[2]))
            rot = self._body_rotations.get(name)
            spin_dir = self._body_spin_dirs.get(name, 1.0)
            if rot is not None:
                if self.align_to_ecliptic:
                    rot = plane_rot @ rot
                mat = np.identity(4, dtype=np.float32)
                mat[:3, :3] = rot
                glMultMatrixf(mat.T)

            texture_id = self._texture_ids.get(name)
            textured = state.available and texture_id is not None
            if textured:
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, texture_id)
                glColor3f(1.0, 1.0, 1.0)
            else:
                glDisable(GL_TEXTURE_2D)
                if state.available:
                    glColor3f(*info.color)
                else:
                    glColor3f(0.4, 0.4, 0.45)

            gluSphere(self._quadric, draw_radius, 18, 18)
            if textured:
                glBindTexture(GL_TEXTURE_2D, 0)
                glDisable(GL_TEXTURE_2D)
            if rot is not None:
                self._draw_rotation_guides(draw_radius, spin_dir, draw_guides=show_guides)
            else:
                self._draw_rotation_guides(draw_radius, spin_dir, draw_guides=show_guides)

            threshold = self._earth_unit_radius() * self.highlight_threshold_earth_units
            if radius < threshold:
                self._draw_bounding_box(max(threshold * 2.0, self.min_visible_radius * 2.0))
            glPopMatrix()

        if self._gears:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            for gear in self._gears.values():
                center = np.array(gear.center_mm, dtype=np.float32)
                glPushMatrix()
                glTranslatef(float(center[0]), float(center[1]), float(center[2]))
                angle_deg = math.degrees(self._gear_angles.get(gear.gear_id, gear.angle_rad))
                glRotatef(angle_deg, 0.0, 1.0, 0.0)
                alpha = 0.85 if gear.is_drive else 0.55
                glColor4f(float(gear.color[0]), float(gear.color[1]), float(gear.color[2]), alpha)
                self._draw_gear_cylinder(gear.radius_mm, gear.thickness_mm)
                glPopMatrix()

                center_screen = self._project_point(center, view, proj)
                self._gear_screen_positions[gear.gear_id] = center_screen
                self._gear_screen_radii[gear.gear_id] = self._screen_radius(
                    center,
                    max(gear.radius_mm, self.min_visible_radius),
                    view,
                    proj,
                    center_screen,
                )

        if self._arms:
            glDisable(GL_BLEND)
            glLineWidth(1.6)
            for arm in self._arms.values():
                angle = float(arm.angle_rad)
                length = float(arm.length_mm)
                origin = np.array(arm.origin_mm, dtype=np.float32)
                x = origin[0] + math.cos(angle) * length
                z = origin[2] + math.sin(angle) * length
                glColor3f(float(arm.color[0]), float(arm.color[1]), float(arm.color[2]))
                glBegin(GL_LINES)
                glVertex3f(float(origin[0]), float(origin[1]), float(origin[2]))
                glVertex3f(float(x), float(origin[1]), float(z))
                glEnd()
                glColor3f(0.45, 0.75, 0.95)
                glBegin(GL_LINES)
                glVertex3f(float(origin[0]), float(arm.tube_base_y_mm), float(origin[2]))
                glVertex3f(float(origin[0]), float(origin[1]), float(origin[2]))
                glEnd()

        if self._mesh_links:
            glDisable(GL_BLEND)
            glLineWidth(1.4)
            glColor3f(0.2, 0.55, 0.95)
            glBegin(GL_LINES)
            for a, b in self._mesh_links:
                glVertex3f(float(a[0]), float(a[1]), float(a[2]))
                glVertex3f(float(b[0]), float(b[1]), float(b[2]))
            glEnd()

        if self.show_system_bounds:
            glColor3f(0.35, 0.7, 0.9)
            for name, radius in self._system_bounds.items():
                center = self._scaled_positions.get(name)
                if center is None:
                    continue
                glPushMatrix()
                glTranslatef(float(center[0]), float(center[1]), float(center[2]))
                self._draw_bounding_box(max(radius * 2.0, self.min_visible_radius * 2.0))
                glPopMatrix()

            glColor3f(0.6, 0.6, 0.6)
            for _name, (inner, outer, _thickness) in self._orbit_sweeps.items():
                if outer <= 0:
                    continue
                glPushMatrix()
                self._draw_bounding_box(max(outer * 2.0, self.min_visible_radius * 2.0))
                glPopMatrix()
                if inner > self.min_visible_radius:
                    glPushMatrix()
                    self._draw_bounding_box(inner * 2.0)
                    glPopMatrix()

        band_alpha = float(self.orbit_band_opacity)
        if self.show_orbit_bands and band_alpha > 0.0:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDepthMask(GL_FALSE)
            for _name, (inner, outer, _thickness) in self._orbit_sweeps.items():
                if outer <= 0 or outer <= inner:
                    continue
                self._draw_orbit_band(inner, outer)
            glDepthMask(GL_TRUE)

        if self.show_orbit_paths and self._orbit_paths:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_LINE_SMOOTH)
            glLineWidth(1.4)
            glEnableClientState(GL_VERTEX_ARRAY)
            for name, path in self._orbit_paths.items():
                if path.size == 0:
                    continue
                settings = self._settings.get(name, BodySettings())
                scaled = np.asarray(path, dtype=np.float32)
                if not self._orbit_paths_scaled:
                    scaled = scaled * distance_scale * settings.distance_scale
                if self.align_to_ecliptic:
                    scaled = (plane_rot @ scaled.T).T
                info = BODY_BY_NAME.get(name)
                if info:
                    glColor4f(float(info.color[0]), float(info.color[1]), float(info.color[2]), 0.7)
                else:
                    glColor4f(0.8, 0.8, 0.8, 0.6)
                center_name = self._orbit_path_centers.get(name)
                if center_name:
                    center = self._scaled_positions.get(center_name)
                else:
                    center = None
                if center is not None:
                    glPushMatrix()
                    glTranslatef(float(center[0]), float(center[1]), float(center[2]))
                arr = np.ascontiguousarray(scaled, dtype=np.float32)
                glVertexPointer(3, GL_FLOAT, 0, arr)
                glDrawArrays(GL_LINE_STRIP, 0, arr.shape[0])
                if center is not None:
                    glPopMatrix()
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisable(GL_LINE_SMOOTH)

        if self.show_moon_bands and self._moon_orbit_center and band_alpha > 0.0:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDepthMask(GL_FALSE)
            center = self._scaled_positions.get(self._moon_orbit_center)
            if center is not None:
                bands = self._moon_orbit_bands.get(self._moon_orbit_center, ())
                for inner, outer in bands:
                    if outer <= 0 or outer <= inner:
                        continue
                    glPushMatrix()
                    glTranslatef(float(center[0]), float(center[1]), float(center[2]))
                    self._draw_orbit_band(inner, outer)
                    glPopMatrix()
            glDepthMask(GL_TRUE)

        self._update_zoom_bounds(body_radii + gear_radii, fov_deg)
        painter.end()
        self._draw_orientation_widget(view)
        if self._gears:
            self._draw_gear_labels()

    def mousePressEvent(self, event) -> None:
        self._last_pos = event.pos()
        self._mouse_press_pos = event.pos()
        self._dragging = False

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.pos() - self._last_pos
            self._yaw += delta.x() * 0.4
            self._pitch += delta.y() * 0.4
            self._pitch = max(-85.0, min(85.0, self._pitch))
            self._last_pos = event.pos()
            if not self._dragging and (event.pos() - self._mouse_press_pos).manhattanLength() > 3:
                self._dragging = True
            self.update()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y() / 120.0
        self._distance *= 0.9 ** delta
        self._distance = max(self._zoom_min, min(self._zoom_max, self._distance))
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:
        target = self._pick_body(event.pos())
        if target:
            self.bodyDoubleClicked.emit(target)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and not self._dragging:
            gear_id = self._pick_gear(event.pos())
            if gear_id:
                self.gearClicked.emit(gear_id)
                event.accept()
                return
        super().mouseReleaseEvent(event)

    def _current_distance_scale(self) -> float:
        if self.auto_distance_scale and self._max_distance_km > 0:
            return self.target_scene_radius / self._max_distance_km
        if self.distance_mode == "mechanical":
            return 1.0
        return self.global_distance_scale

    def _compute_max_distance(self) -> float:
        distances = []
        for state in self._positions.values():
            if not state.available:
                continue
            vec = np.array(state.position_km, dtype=np.float32)
            distances.append(float(np.linalg.norm(vec)))
        return max(distances, default=0.0)

    def _camera_eye(self) -> np.ndarray:
        yaw_rad = math.radians(self._yaw)
        pitch_rad = math.radians(self._pitch)
        x = self._distance * math.cos(pitch_rad) * math.cos(yaw_rad)
        y = self._distance * math.sin(pitch_rad)
        z = self._distance * math.cos(pitch_rad) * math.sin(yaw_rad)
        return self._target + np.array([x, y, z], dtype=np.float32)

    def _project_point(
        self,
        point: np.ndarray,
        view: np.ndarray,
        proj: np.ndarray,
    ) -> Tuple[float, float, float]:
        clip = proj @ view @ np.array([point[0], point[1], point[2], 1.0], dtype=np.float32)
        if clip[3] == 0:
            return (0.0, 0.0, 1.0)
        ndc = clip[:3] / clip[3]
        x = (ndc[0] * 0.5 + 0.5) * self.width()
        y = (1.0 - (ndc[1] * 0.5 + 0.5)) * self.height()
        return (float(x), float(y), float(ndc[2]))

    def _pick_body(self, pos: QPoint) -> Optional[str]:
        closest_name = None
        closest_dist = 1.0e9
        for name, (x, y, _z) in self._screen_positions.items():
            dx = x - pos.x()
            dy = y - pos.y()
            dist = math.hypot(dx, dy)
            if dist < 16.0 and dist < closest_dist:
                closest_dist = dist
                closest_name = name
        return closest_name

    def _pick_gear(self, pos: QPoint) -> Optional[str]:
        closest_id = None
        closest_dist = 1.0e9
        for gear_id, (x, y, _z) in self._gear_screen_positions.items():
            dx = x - pos.x()
            dy = y - pos.y()
            dist = math.hypot(dx, dy)
            radius = self._gear_screen_radii.get(gear_id, 10.0)
            threshold = max(10.0, radius)
            if dist < threshold and dist < closest_dist:
                closest_dist = dist
                closest_id = gear_id
        return closest_id

    def _scaled_radius(self, radius_km: float, body_scale: float) -> float:
        if radius_km <= 0:
            return self.min_visible_radius
        if self.size_mode == "log":
            ref = self.reference_radius_km if self.reference_radius_km > 0 else 1.0
            ratio = radius_km / ref
            if ratio <= 0:
                return self.min_visible_radius
            value = math.log(ratio, self.size_log_base)
            diameter_mm = self.earth_diameter_mm * (1.0 + value)
            diameter_mm = max(diameter_mm, self.min_body_diameter_mm)
            return (diameter_mm / 2.0) * body_scale
        return radius_km * self.global_radius_scale * body_scale

    def _earth_unit_radius(self) -> float:
        return max(self.earth_diameter_mm / 2.0, self.min_visible_radius)

    def _draw_bounding_box(self, size: float) -> None:
        half = size / 2.0
        glLineWidth(1.2)
        glBegin(GL_LINE_LOOP)
        glVertex3f(-half, -half, -half)
        glVertex3f(half, -half, -half)
        glVertex3f(half, half, -half)
        glVertex3f(-half, half, -half)
        glEnd()

        glBegin(GL_LINE_LOOP)
        glVertex3f(-half, -half, half)
        glVertex3f(half, -half, half)
        glVertex3f(half, half, half)
        glVertex3f(-half, half, half)
        glEnd()

        glBegin(GL_LINES)
        glVertex3f(-half, -half, -half)
        glVertex3f(-half, -half, half)
        glVertex3f(half, -half, -half)
        glVertex3f(half, -half, half)
        glVertex3f(half, half, -half)
        glVertex3f(half, half, half)
        glVertex3f(-half, half, -half)
        glVertex3f(-half, half, half)
        glEnd()

    def _draw_gear_cylinder(self, radius: float, height: float) -> None:
        if self._quadric is None:
            return
        h = max(float(height), 0.01)
        r = max(float(radius), 0.01)
        glPushMatrix()
        glRotatef(-90.0, 1.0, 0.0, 0.0)
        glTranslatef(0.0, 0.0, -h * 0.5)
        gluCylinder(self._quadric, r, r, h, 24, 1)
        gluDisk(self._quadric, 0.0, r, 24, 1)
        glTranslatef(0.0, 0.0, h)
        gluDisk(self._quadric, 0.0, r, 24, 1)
        glColor3f(0.0, 0.0, 0.0)
        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(r * 0.92, 0.0, 0.0)
        glEnd()
        glPopMatrix()

    def _draw_orbit_band(self, inner: float, outer: float) -> None:
        segments = 128
        alpha = max(0.0, min(1.0, float(self.orbit_band_opacity)))
        glColor4f(0.25, 0.55, 0.75, alpha)

        glBegin(GL_TRIANGLE_STRIP)
        for i in range(segments + 1):
            angle = 2.0 * math.pi * (i / segments)
            c = math.cos(angle)
            s = math.sin(angle)
            inner_vec = np.array([inner * c, 0.0, inner * s], dtype=np.float32)
            outer_vec = np.array([outer * c, 0.0, outer * s], dtype=np.float32)
            glVertex3f(float(outer_vec[0]), float(outer_vec[1]), float(outer_vec[2]))
            glVertex3f(float(inner_vec[0]), float(inner_vec[1]), float(inner_vec[2]))
        glEnd()

    def _compute_clip_planes(
        self,
        body_radii,
        fov_deg: float,
        camera_distance: float,
        distance_scale: float,
    ) -> Tuple[float, float]:
        scene_radius = 0.0
        for name, pos in self._scaled_positions.items():
            radius = 0.0
            if name in BODY_BY_NAME:
                radius = max(self._scaled_radius(BODY_BY_NAME[name].radius_km, 1.0), 0.0)
            scene_radius = max(scene_radius, float(np.linalg.norm(pos)) + radius)

        for name, radius in self._system_bounds.items():
            center = self._scaled_positions.get(name)
            if center is None:
                continue
            scene_radius = max(scene_radius, float(np.linalg.norm(center)) + radius)

        for _name, (_inner, outer, thickness) in self._orbit_sweeps.items():
            scene_radius = max(scene_radius, outer + thickness)
        if self.show_orbit_paths and self._orbit_paths:
            scene_radius = max(scene_radius, self._orbit_path_radius(distance_scale))
        for gear in self._gears.values():
            center = np.array(gear.center_mm, dtype=np.float32)
            scene_radius = max(scene_radius, float(np.linalg.norm(center)) + gear.radius_mm)
        for arm in self._arms.values():
            origin = np.array(arm.origin_mm, dtype=np.float32)
            arm_extent = math.hypot(arm.length_mm, origin[1])
            scene_radius = max(scene_radius, float(np.linalg.norm(origin)) + arm_extent)

        if scene_radius <= 0:
            scene_radius = 1.0
        smallest = min(body_radii) if body_radii else self.min_visible_radius
        smallest = max(smallest, self.min_visible_radius if not self.real_scale_mode else 0.0)
        distance = max(camera_distance, 1.0e-6)
        far_margin = max(scene_radius * 1.6, smallest * 4.0, 1.0e-6)
        near_margin = max(min(far_margin, distance * 0.9), smallest * 4.0, 1.0e-6)
        far = max(distance + far_margin, scene_radius * 2.0, 10.0)
        near = distance - near_margin
        if near <= 0:
            min_near = 1.0e-6 if self.real_scale_mode else 1.0e-3
            near = max(smallest * 0.05, distance * 1.0e-4, min_near)
        return (near, far)

    def _draw_orientation_widget(self, view: np.ndarray) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        size = 84
        margin = 10
        left = self.width() - size - margin
        top = margin
        center = QPoint(left + size // 2, top + size // 2)

        painter.setPen(QPen(QColor(230, 230, 230), 1))
        painter.setBrush(QColor(30, 30, 40, 180))
        painter.drawRoundedRect(left, top, size, size, 6, 6)

        axes = {
            "X": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "Y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "Z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        colors = {
            "X": QColor(231, 76, 60),
            "Y": QColor(46, 204, 113),
            "Z": QColor(52, 152, 219),
        }

        rot = view[:3, :3]
        scale = size * 0.32
        painter.setFont(QFont("Segoe UI", 8))

        for label, axis in axes.items():
            cam = rot @ axis
            end = QPoint(int(center.x() + cam[0] * scale), int(center.y() - cam[1] * scale))
            painter.setPen(QPen(colors[label], 2))
            painter.drawLine(center, end)
            painter.drawText(end + QPoint(2, -2), label)

        painter.setPen(QPen(QColor(220, 220, 220), 1))
        label = "UP=+Y (Orbital)" if self._orbit_normal is not None else "UP=+Y (Ecliptic)"
        painter.drawText(left + 6, top + size - 6, label)
        painter.end()

    def _draw_gear_labels(self) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setFont(QFont("Segoe UI", 8))
        for gear in self._gears.values():
            screen = self._gear_screen_positions.get(gear.gear_id)
            if screen is None:
                continue
            x, y, _z = screen
            radius = self._gear_screen_radii.get(gear.gear_id, 10.0)
            color = QColor(220, 80, 80) if gear.is_drive else QColor(210, 210, 210)
            painter.setPen(QPen(color))
            painter.drawText(int(x + radius * 0.2), int(y - radius * 0.2), str(gear.teeth))
        painter.end()

    def _draw_rotation_guides(
        self,
        radius: float,
        spin_dir: float = 1.0,
        draw_guides: bool = True,
    ) -> None:
        if not draw_guides:
            return
        axis_len = radius * 1.25
        glLineWidth(1.6)
        glColor3f(0.9, 0.15, 0.15)
        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, -axis_len)
        glVertex3f(0.0, 0.0, axis_len)
        glEnd()

        ring_radius = radius * 1.1
        segments = 48
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = 2.0 * math.pi * (i / segments)
            glVertex3f(float(math.cos(angle) * ring_radius), float(math.sin(angle) * ring_radius), 0.0)
        glEnd()

        if draw_guides:
            arrow_angle = 0.0
            arrow_tip = np.array(
                [math.cos(arrow_angle) * ring_radius, math.sin(arrow_angle) * ring_radius, 0.0],
                dtype=np.float32,
            )
            spin = 1.0 if spin_dir >= 0.0 else -1.0
            tangent = np.array(
                [-math.sin(arrow_angle), math.cos(arrow_angle), 0.0],
                dtype=np.float32,
            ) * spin
            arrow_len = ring_radius * 0.18
            arrow_wing = ring_radius * 0.08
            base = arrow_tip - tangent * arrow_len
            wing_offset = np.array(
                [math.cos(arrow_angle) * arrow_wing, math.sin(arrow_angle) * arrow_wing, 0.0],
                dtype=np.float32,
            )
            wing_left = base + wing_offset
            wing_right = base - wing_offset
            glBegin(GL_LINES)
            glVertex3f(float(arrow_tip[0]), float(arrow_tip[1]), float(arrow_tip[2]))
            glVertex3f(float(wing_left[0]), float(wing_left[1]), float(wing_left[2]))
            glVertex3f(float(arrow_tip[0]), float(arrow_tip[1]), float(arrow_tip[2]))
            glVertex3f(float(wing_right[0]), float(wing_right[1]), float(wing_right[2]))
            glEnd()

        glLineWidth(1.4)
        glColor3f(0.0, 0.0, 0.0)
        glBegin(GL_LINE_STRIP)
        steps = 32
        meridian_radius = radius * 1.02
        for i in range(steps + 1):
            theta = -0.5 * math.pi + math.pi * (i / steps)
            x = meridian_radius * math.cos(theta)
            z = meridian_radius * math.sin(theta)
            glVertex3f(float(x), 0.0, float(z))
        glEnd()

    def _screen_radius(
        self,
        center: np.ndarray,
        radius: float,
        view: np.ndarray,
        proj: np.ndarray,
        center_screen: Tuple[float, float, float],
    ) -> float:
        edge_world = center + np.array([radius, 0.0, 0.0], dtype=np.float32)
        edge_screen = self._project_point(edge_world, view, proj)
        return float(math.hypot(edge_screen[0] - center_screen[0], edge_screen[1] - center_screen[1]))

    def _orbital_plane_rotation(self) -> np.ndarray:
        if not self.align_to_ecliptic:
            return np.identity(3, dtype=np.float32)
        if self._orbit_normal is not None:
            target = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            n = self._orbit_normal
            dot = float(np.clip(np.dot(n, target), -1.0, 1.0))
            if abs(dot - 1.0) < 1.0e-6:
                return np.identity(3, dtype=np.float32)
            if abs(dot + 1.0) < 1.0e-6:
                # 180 degree flip around X axis
                return np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, -1.0],
                    ],
                    dtype=np.float32,
                )
            axis = np.cross(n, target)
            axis_norm = np.linalg.norm(axis)
            if axis_norm <= 1.0e-6:
                return np.identity(3, dtype=np.float32)
            axis = axis / axis_norm
            angle = math.acos(dot)
            c = math.cos(angle)
            s = math.sin(angle)
            t = 1.0 - c
            x, y, z = axis
            return np.array(
                [
                    [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                    [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                    [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
                ],
                dtype=np.float32,
            )

        # Fallback: fixed ecliptic tilt (J2000)
        angle = math.radians(self.ecliptic_tilt_deg - 90.0)
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, c, -s],
                [0.0, s, c],
            ],
            dtype=np.float32,
        )

    def _update_zoom_bounds(self, body_radii, fov_deg: float) -> None:
        half_fov = math.radians(fov_deg / 2.0)
        tan_half = math.tan(half_fov)
        if tan_half <= 0:
            return

        scene_radius = 0.0
        for name, pos in self._scaled_positions.items():
            radius = 0.0
            if name in BODY_BY_NAME:
                radius = max(self._scaled_radius(BODY_BY_NAME[name].radius_km, 1.0), 0.0)
            scene_radius = max(scene_radius, float(np.linalg.norm(pos)) + radius)

        for name, radius in self._system_bounds.items():
            center = self._scaled_positions.get(name)
            if center is None:
                continue
            scene_radius = max(scene_radius, float(np.linalg.norm(center)) + radius)

        for _name, (_inner, outer, thickness) in self._orbit_sweeps.items():
            scene_radius = max(scene_radius, outer + thickness)
        if self.show_orbit_paths and self._orbit_paths:
            scene_radius = max(scene_radius, self._orbit_path_radius(self._current_distance_scale()))
        for gear in self._gears.values():
            center = np.array(gear.center_mm, dtype=np.float32)
            scene_radius = max(scene_radius, float(np.linalg.norm(center)) + gear.radius_mm)
        for arm in self._arms.values():
            origin = np.array(arm.origin_mm, dtype=np.float32)
            arm_extent = math.hypot(arm.length_mm, origin[1])
            scene_radius = max(scene_radius, float(np.linalg.norm(origin)) + arm_extent)

        if scene_radius <= 0:
            scene_radius = 1.0

        smallest_radius = min(body_radii) if body_radii else self.min_visible_radius
        smallest_radius = max(smallest_radius, self.min_visible_radius)

        min_limit = 0.0 if self.real_scale_mode else 1.0
        self._zoom_max = max(scene_radius / tan_half * 1.1, 10.0)
        self._zoom_min = max(smallest_radius / tan_half * 0.8, min_limit)
        if self._zoom_min > self._zoom_max:
            self._zoom_min = max(self._zoom_max * 0.2, min_limit)

    def _orbit_path_radius(self, distance_scale: float) -> float:
        max_radius = 0.0
        for name, path in self._orbit_paths.items():
            if path.size == 0:
                continue
            settings = self._settings.get(name, BodySettings())
            scale = 1.0 if self._orbit_paths_scaled else distance_scale * settings.distance_scale
            center_name = self._orbit_path_centers.get(name)
            center = self._scaled_positions.get(center_name) if center_name else None
            if center is None:
                radii = np.linalg.norm(path, axis=1)
                max_radius = max(max_radius, float(np.max(radii)) * scale)
            else:
                scaled = np.asarray(path, dtype=float) * scale
                radii = np.linalg.norm(scaled + center, axis=1)
                max_radius = max(max_radius, float(np.max(radii)))
        return max_radius
