"""Cooperative GPU breathing room for training on a desktop display GPU."""
import math
import os
import time

import torch


class GpuPacer:
    def __init__(self, device, fraction=None):
        self.fraction = float(os.environ.get('MICROGPT_GPU_DUTY_CYCLE', '0.75')) if fraction is None else fraction
        if not math.isfinite(self.fraction) or not 0 < self.fraction <= 1:
            raise ValueError('MICROGPT_GPU_DUTY_CYCLE must be in (0, 1]')
        self.full_until = float(os.environ.get("MICROGPT_FULL_GPU_UNTIL", "0")) if fraction is None else 0.0
        if not math.isfinite(self.full_until):
            raise ValueError("MICROGPT_FULL_GPU_UNTIL must be a finite Unix timestamp")
        self.device = device
        self.enabled = device.type == 'cuda' and self.fraction < 1
        self.last = time.monotonic()

    def wait(self):
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
