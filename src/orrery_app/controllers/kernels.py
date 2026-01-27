from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox

from ..spice_service import SpiceEphemeris

if TYPE_CHECKING:
    _MainWindowBase = Any
else:
    _MainWindowBase = object


class KernelLoadWorker(QObject):
    finished = Signal(str, object, object)

    def __init__(self, spice: SpiceEphemeris, kernel_dir: Path) -> None:
        super().__init__()
        self._spice = spice
        self._kernel_dir = kernel_dir

    def run(self) -> None:
        try:
            result = self._spice.load_kernels(self._kernel_dir)
            status = self._spice.status_message
            error = None
        except Exception as exc:  # noqa: BLE001
            result = None
            status = "Kernel load failed"
            error = exc
        self.finished.emit(status, result, error)


class KernelControlMixin(_MainWindowBase):
    def _start_kernel_load(self) -> None:
        self._set_loading_step("Loading SPICE kernels...")
        self._kernel_thread = QThread(self)
        self._kernel_worker = KernelLoadWorker(self.spice, self.spice.kernel_dir)
        self._kernel_worker.moveToThread(self._kernel_thread)
        self._kernel_thread.started.connect(self._kernel_worker.run)
        self._kernel_worker.finished.connect(self._finish_kernel_load)
        self._kernel_worker.finished.connect(self._kernel_thread.quit)
        self._kernel_worker.finished.connect(self._kernel_worker.deleteLater)
        self._kernel_thread.finished.connect(self._kernel_thread.deleteLater)
        self._kernel_thread.start()

    def _finish_kernel_load(self, status_message: str, result, error) -> None:
        self.kernel_status.setText(f"Status: {status_message}")
        if error is not None:
            QMessageBox.critical(
                self,
                "Kernel load failed",
                f"Kernel load failed:\n{error}",
            )
        elif result and getattr(result, "errors", None):
            msg = "Some kernels failed to load:\n" + "\n".join(result.errors[:6])
            if len(result.errors) > 6:
                msg += f"\n...and {len(result.errors) - 6} more"
            QMessageBox.warning(self, "Kernel load issues", msg)
        self._set_loading_step("Preparing layout...")
        self._apply_mechanical_settings()
        self._kernels_loaded = True
        self._finalize_loading_if_ready()

    def _browse_kernel_dir(self) -> None:
        start_dir = self.kernel_path.text().strip() or str(self.spice.kernel_dir)
        selected = QFileDialog.getExistingDirectory(self, "Select kernel folder", start_dir)
        if selected:
            self.kernel_path.setText(selected)

    def _load_kernels(self) -> None:
        kernel_dir = Path(self.kernel_path.text().strip())
        result = self.spice.load_kernels(kernel_dir)
        self.kernel_status.setText(f"Status: {self.spice.status_message}")
        if result.errors:
            msg = "Some kernels failed to load:\n" + "\n".join(result.errors[:6])
            if len(result.errors) > 6:
                msg += f"\n...and {len(result.errors) - 6} more"
            QMessageBox.warning(self, "Kernel load issues", msg)
        self._refresh_positions()
        self._update_diagnostics()
