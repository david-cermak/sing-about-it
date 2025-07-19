"""Logging and progress tracking utilities."""

import time
from typing import Optional
from datetime import datetime
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings


class Timer:
    """Simple timer for tracking elapsed time."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        self.end_time = None

    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if self.start_time is None:
            return 0.0
        self.end_time = time.time()
        return self.elapsed()

    def elapsed(self) -> float:
        """Get elapsed time (whether timer is stopped or not)."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time


def log_info(message: str, verbose_only: bool = False):
    """Log an info message."""
    if not verbose_only or settings.logging.verbose_output:
        print(f"ℹ️  {message}")


def log_success(message: str, verbose_only: bool = False):
    """Log a success message."""
    if not verbose_only or settings.logging.verbose_output:
        print(f"✅ {message}")


def log_warning(message: str, verbose_only: bool = False):
    """Log a warning message."""
    if not verbose_only or settings.logging.verbose_output:
        print(f"⚠️  {message}")


def log_error(message: str, verbose_only: bool = False):
    """Log an error message."""
    if not verbose_only or settings.logging.verbose_output:
        print(f"❌ {message}")


def log_progress(current: int, total: int, task: str):
    """Log progress for a task."""
    if settings.logging.verbose_output:
        percentage = (current / total) * 100 if total > 0 else 0
        print(f"🔄 {task}: {current}/{total} ({percentage:.1f}%)")


def format_duration(seconds: float) -> str:
    """Format duration in a human-readable way."""
    if seconds < 1:
        return f"{seconds:.2f}s"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"


def log_phase_start(phase_name: str):
    """Log the start of a processing phase."""
    if settings.logging.verbose_output:
        print(f"\n🚀 Starting {phase_name}")
        print("=" * 60)


def log_phase_end(phase_name: str, duration: float):
    """Log the end of a processing phase."""
    if settings.logging.verbose_output:
        print(f"✅ Completed {phase_name} in {format_duration(duration)}")
        print("=" * 60)
