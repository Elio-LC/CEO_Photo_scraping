"""Progress tracking utilities."""

from __future__ import annotations

import time


class ProgressTracker:
    """Progress tracker with elapsed/ETA reporting."""

    def __init__(self, total: int, prefix: str = "Processing"):
        self.total = total
        self.current = 0
        self.prefix = prefix
        self.start_time = time.time()

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def update(self, current: int, suffix: str = "") -> None:
        """Update progress and show time estimates."""
        self.current = current
        percentage = (current / self.total * 100) if self.total > 0 else 0
        bar_length = 40
        filled = int(bar_length * current / self.total) if self.total > 0 else 0
        bar = "█" * filled + "░" * (bar_length - filled)

        elapsed = time.time() - self.start_time
        if current > 0 and self.total > 0:
            avg_per_item = elapsed / current
            total_estimated = avg_per_item * self.total
            remaining = total_estimated - elapsed
            total_str = self._format_duration(total_estimated)
            remaining_str = self._format_duration(remaining)
        else:
            total_estimated = 0
            remaining = 0
            total_str = "--:--"
            remaining_str = "--:--"

        elapsed_str = self._format_duration(elapsed)
        time_suffix = (
            f"elapsed {elapsed_str} / total {total_str} | remaining {remaining_str}"
        )
        combined_suffix = f"{time_suffix} {suffix}".strip()

        print(
            f"\r{self.prefix} [{bar}] {percentage:.1f}% ({current}/{self.total}) {combined_suffix}",
            end="",
        )

    def finish(self) -> None:
        """Mark progress as finished."""
        self.update(self.total, "Done!")
        print()
