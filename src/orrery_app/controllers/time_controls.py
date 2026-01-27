from __future__ import annotations

from datetime import timedelta
from time import perf_counter
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QDateTime, Qt

if TYPE_CHECKING:
    _MainWindowBase = Any
else:
    _MainWindowBase = object


class TimeControlMixin(_MainWindowBase):
    def _set_now(self) -> None:
        self.time_edit.setDateTime(QDateTime.currentDateTimeUtc())
        self._refresh_positions()

    def _set_time_speed(self, speed: float) -> None:
        clamped = max(0.0, float(speed))
        if clamped <= 0.0:
            self._pause_time()
            return

        self._time_speed = clamped
        self._last_nonzero_speed = clamped
        self._time_last_tick = perf_counter()
        if not self._time_timer.isActive():
            self._time_timer.start()
        self.time_overlay.set_running(True)
        self.time_overlay.set_speed(clamped)

    def _toggle_time_playback(self, running: bool) -> None:
        if running:
            speed = self._last_nonzero_speed if self._last_nonzero_speed > 0 else 86400.0
            self._set_time_speed(speed)
        else:
            self._pause_time()

    def _pause_time(self) -> None:
        self._time_speed = 0.0
        self._time_last_tick = None
        if self._time_timer.isActive():
            self._time_timer.stop()
        self.time_overlay.set_running(False)
        self.time_overlay.set_speed(0.0)

    def _advance_time(self) -> None:
        if self._time_speed <= 0.0:
            return
        now = perf_counter()
        if self._time_last_tick is None:
            self._time_last_tick = now
            return
        delta = now - self._time_last_tick
        self._time_last_tick = now
        if delta <= 0:
            return
        self._advance_gear_animation(delta)
        current = self._current_datetime()
        next_time = current + timedelta(seconds=delta * self._time_speed)
        ms = int(next_time.timestamp() * 1000)
        self.time_edit.setDateTime(QDateTime.fromMSecsSinceEpoch(ms, Qt.TimeSpec.UTC))
        self._refresh_positions(refresh_orbit_stats=False, update_diagnostics=False)
