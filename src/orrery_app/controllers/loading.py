from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

if TYPE_CHECKING:
    _MainWindowBase = Any
else:
    _MainWindowBase = object


class LoadingMixin(_MainWindowBase):
    def _maybe_start_initial_load(self) -> None:
        if self._did_initial_load:
            return
        self._did_initial_load = True
        self._show_loading_overlay(True)
        self._start_textures_if_ready()
        QTimer.singleShot(0, self._start_kernel_load)

    def _mark_gl_ready(self) -> None:
        self._gl_ready = True
        self._start_textures_if_ready()

    def _start_textures_if_ready(self) -> None:
        if not self._gl_ready or self._textures_started:
            return
        self._textures_started = True
        QTimer.singleShot(0, self.gl_widget.load_textures)

    def _show_loading_overlay(self, visible: bool) -> None:
        self._loading_overlay.setVisible(visible)
        if visible:
            self._loading_overlay.raise_()
            self._loading_overlay.setGeometry(self.centralWidget().rect())

    def _set_loading_step(self, text: str) -> None:
        if hasattr(self, "_loading_step_label"):
            self._loading_step_label.setText(text)
            QApplication.processEvents()

    def _mark_textures_loaded(self) -> None:
        self._textures_loaded = True
        self._finalize_loading_if_ready()

    def _finalize_loading_if_ready(self) -> None:
        if self._textures_loaded and self._kernels_loaded:
            self._show_loading_overlay(False)
