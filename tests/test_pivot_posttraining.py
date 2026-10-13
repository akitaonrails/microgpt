"""Recovery must never replay post-training work or change the holdout."""
import unittest

from scripts.run_pivot_posttraining import validate_waiting_resume


class WaitingResumeTests(unittest.TestCase):
    def state(self, **changes):
        return dict(status='failed', commands=[], holdout_sha256='original',
                    error='Foundation pipeline requires recovery before further GPU work') | changes

    def test_upstream_interruption_and_manual_pause_can_resume(self):
        validate_waiting_resume(self.state(), 'original')
        validate_waiting_resume(self.state(status='interrupted'), 'original')

    def test_started_work_other_failures_and_active_queue_are_rejected(self):
        for changes in ({'commands': [['train']]}, {'error': 'Different failure'},
                        {'status': 'waiting_for_foundation'}, {'status': 'complete_no_agent_readiness_claim'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_waiting_resume(self.state(**changes), 'original')

    def test_changed_holdout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Holdout changed'):
            validate_waiting_resume(self.state(), 'changed')
