"""Pacing drains queued work before yielding; it never changes training math."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from microgpt.pacing import GpuPacer


class PacingTests(unittest.TestCase):
    def test_cuda_work_finishes_before_idle_interval(self):
        events = []
        with patch('microgpt.pacing.time.monotonic', side_effect=[0, .12, .16, .28, .32]), \
             patch('microgpt.pacing.torch.cuda.synchronize', side_effect=lambda _: events.append('sync')), \
             patch('microgpt.pacing.time.sleep', side_effect=lambda seconds: events.append(seconds)):
            pacer = GpuPacer(SimpleNamespace(type='cuda'), .75)
            pacer.wait(); pacer.wait()
        self.assertEqual(events[0], 'sync')
        self.assertAlmostEqual(events[1], .04)
        self.assertEqual(events[2], 'sync')
        self.assertAlmostEqual(events[3], .04)

    def test_full_gpu_permission_expires_during_run(self):
        with patch.dict('os.environ', {'MICROGPT_GPU_DUTY_CYCLE': '1', 'MICROGPT_FULL_GPU_UNTIL': '100'}), \
             patch('microgpt.pacing.time.time', side_effect=[99, 101]), \
             patch('microgpt.pacing.torch.cuda.synchronize') as sync, \
             patch('microgpt.pacing.time.sleep') as sleep:
            pacer = GpuPacer(SimpleNamespace(type='cuda'))
            pacer.wait()
            sync.assert_not_called()
            pacer.wait()
            sync.assert_called_once()
            sleep.assert_called_once()
            self.assertEqual(pacer.fraction, .75)

    def test_unrestricted_and_cpu_modes_do_not_sleep_or_synchronize(self):
        with patch('microgpt.pacing.torch.cuda.synchronize') as sync, patch('microgpt.pacing.time.sleep') as sleep:
            GpuPacer(SimpleNamespace(type='cpu'), .75).wait()
            GpuPacer(SimpleNamespace(type='cuda'), 1).wait()
        sync.assert_not_called(); sleep.assert_not_called()
        for fraction in (0, -1, 2, float('nan')):
            with self.assertRaises(ValueError):
                GpuPacer(SimpleNamespace(type='cuda'), fraction)
