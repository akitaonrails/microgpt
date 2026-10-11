"""Cooperative GPU breathing room for training on a desktop display GPU."""
from datetime import datetime
from zoneinfo import ZoneInfo
import math
import os
import time

import torch


def scheduled_fraction(timestamp, window, timezone):
    """Start inclusive, end exclusive; handles a window crossing midnight."""
    start, end = window
    now = datetime.fromtimestamp(timestamp, timezone)
    minute = now.hour * 60 + now.minute
    active = start <= minute < end if start < end else minute >= start or minute < end
    return 1.0 if active else 0.75


def parse_window(value):
    try:
        parts = [tuple(map(int, part.split(':'))) for part in value.split('-')]
        if len(parts) != 2 or any(len(p) != 2 or not 0 <= p[0] < 24 or not 0 <= p[1] < 60 for p in parts):
            raise ValueError
        result = tuple(hour * 60 + minute for hour, minute in parts)
        if result[0] == result[1]: raise ValueError
        return result
    except (ValueError, TypeError):
        raise ValueError('MICROGPT_FULL_GPU_WINDOW must be distinct HH:MM-HH:MM times') from None


class GpuPacer:
    def __init__(self, device, fraction=None):
        self.fraction = float(os.environ.get('MICROGPT_GPU_DUTY_CYCLE', '0.75')) if fraction is None else fraction
        if not math.isfinite(self.fraction) or not 0 < self.fraction <= 1:
            raise ValueError('MICROGPT_GPU_DUTY_CYCLE must be in (0, 1]')
        self.full_until = float(os.environ.get("MICROGPT_FULL_GPU_UNTIL", "0")) if fraction is None else 0.0
        if not math.isfinite(self.full_until):
            raise ValueError("MICROGPT_FULL_GPU_UNTIL must be a finite Unix timestamp")
        window = os.environ.get('MICROGPT_FULL_GPU_WINDOW') if fraction is None else None
        self.window = parse_window(window) if window else None
        self.timezone = ZoneInfo(os.environ.get('MICROGPT_GPU_TIMEZONE', 'America/Sao_Paulo')) if self.window else None
        if self.window:
            self.full_until = 0.0
            self.fraction = scheduled_fraction(time.time(), self.window, self.timezone)
        self.device = device
        self.enabled = device.type == 'cuda' and self.fraction < 1
        self.last = time.monotonic()

    def wait(self):
        if self.window:
            fraction = scheduled_fraction(time.time(), self.window, self.timezone)
            if fraction != self.fraction:
                self.fraction = fraction
                self.enabled = self.device.type == 'cuda' and fraction < 1
                self.last = time.monotonic()
        if self.full_until and time.time() >= self.full_until:
            self.fraction = min(self.fraction, 0.75)
            self.enabled = self.device.type == "cuda"
            self.full_until = 0.0
        if not self.enabled:
            return
        # Host sleeps alone are insufficient: previously queued CUDA work must
        # finish first. This is a soft duty cycle, not a hardware reservation.
        torch.cuda.synchronize(self.device)
        elapsed = time.monotonic() - self.last
        time.sleep(min(elapsed, 1.0) * (1 - self.fraction) / self.fraction)
        self.last = time.monotonic()
