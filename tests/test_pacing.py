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

    def test_daily_window_boundaries_and_midnight(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from microgpt.pacing import scheduled_fraction, parse_window
        zone = ZoneInfo('America/Sao_Paulo')
        window = parse_window('03:00-09:00')
        for hour, minute, expected in [(2,59,.75),(3,0,1),(8,59,1),(9,0,.75),(23,59,.75)]:
            timestamp = datetime(2026,10,11,hour,minute,tzinfo=zone).timestamp()
            self.assertEqual(scheduled_fraction(timestamp,window,zone), expected)
        for hour, expected in [(23,1),(0,1),(3,.75)]:
            self.assertEqual(scheduled_fraction(datetime(2026,10,11,hour,tzinfo=zone).timestamp(),parse_window('22:00-03:00'),zone),expected)
        for value in ('24:00-09:00','03:99-09:00','03:00-03:00','bad'):
            with self.assertRaises(ValueError): parse_window(value)

    def test_daily_schedule_changes_without_restarting(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        zone=ZoneInfo('America/Sao_Paulo')
        stamps=[datetime(2026,10,11,h,tzinfo=zone).timestamp() for h in (2,3,9)]
        with patch.dict('os.environ', {'MICROGPT_FULL_GPU_WINDOW':'03:00-09:00','MICROGPT_GPU_TIMEZONE':'America/Sao_Paulo'}), \
             patch('microgpt.pacing.time.time', side_effect=stamps), \
             patch('microgpt.pacing.torch.cuda.synchronize') as sync, \
             patch('microgpt.pacing.time.sleep'):
            pacer=GpuPacer(SimpleNamespace(type='cuda'))
            self.assertEqual(pacer.fraction,.75)
            pacer.wait();self.assertEqual(pacer.fraction,1)
            sync.assert_not_called()
            pacer.wait();self.assertEqual(pacer.fraction,.75)
            sync.assert_called_once()
