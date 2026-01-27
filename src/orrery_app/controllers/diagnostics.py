from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    _MainWindowBase = Any
else:
    _MainWindowBase = object


class DiagnosticsMixin(_MainWindowBase):
    def _update_diagnostics(self) -> None:
        lines = []
        lines.append(f"SPICE: {self.spice.status_message}")
        lines.append(f"Kernel dir: {self.kernel_path.text().strip()}")

        if self.spice.load_errors:
            lines.append("Kernel load errors:")
            for err in self.spice.load_errors[:6]:
                lines.append(f"  - {err}")
            if len(self.spice.load_errors) > 6:
                lines.append(f"  ...and {len(self.spice.load_errors) - 6} more")

        missing_names = [name for name, state in self._positions.items() if not state.available]
        if missing_names:
            lines.append(f"Missing bodies ({len(missing_names)}): {', '.join(missing_names[:12])}")
            if len(missing_names) > 12:
                lines.append(f"...and {len(missing_names) - 12} more")
            error_counts = {}
            for name in missing_names:
                state = self._positions.get(name)
                err = state.error if state else None
                if not err:
                    continue
                error_counts[err] = error_counts.get(err, 0) + 1
            if error_counts:
                lines.append("Sample SPICE errors:")
                for err, count in list(error_counts.items())[:6]:
                    lines.append(f"  - ({count}x) {err}")
        else:
            lines.append("Missing bodies: 0")

        try:
            kernel_dir = Path(self.kernel_path.text().strip())
            tmp_files = list(kernel_dir.rglob("BIT*.tmp"))
            if tmp_files:
                lines.append("Warning: kernel downloads still in progress (.tmp files present).")
        except Exception:
            pass

        self.diagnostics.setPlainText("\n".join(lines))
