"""Agent success needs a correct artifact, a successful check and a clean finish."""
import unittest
from scripts.eval_opencode import completion_evidence


class AgentCompletionTests(unittest.TestCase):
    def check(self, output='PASS\n', tool='check'):
        return {'type': 'tool_use', 'part': {'tool': tool, 'state': {'status': 'completed', 'output': output}}}

    def test_correct_file_without_check_is_not_verified(self):
        events = [{'type': 'step_finish', 'part': {'reason': 'stop'}}]
        result = completion_evidence(events, True, 0)
        self.assertTrue(result['code_correct'])
        self.assertTrue(result['completed'])
        self.assertFalse(result['passed'])

    def test_recovery_then_success_is_allowed(self):
        events = [self.check('FAIL\n'), self.check(), {'type': 'step_finish', 'part': {'reason': 'stop'}}]
        self.assertTrue(completion_evidence(events, True, 0)['passed'])
        self.assertFalse(completion_evidence(events, False, 0)['passed'])

    def test_error_or_exhausted_run_is_not_complete(self):
        for ending, returncode in [({'type': 'error', 'error': 'failed'}, 0),
                                   ({'type': 'step_finish', 'part': {'reason': 'length'}}, 0),
                                   ({'type': 'step_finish', 'part': {'reason': 'stop'}}, 1)]:
            with self.subTest(ending=ending, returncode=returncode):
                self.assertFalse(completion_evidence([self.check(), ending], True, returncode)['passed'])

    def test_pass_text_from_other_tool_is_not_a_check(self):
        events = [self.check(tool='read'), {'type': 'step_finish', 'part': {'reason': 'stop'}}]
        self.assertFalse(completion_evidence(events, True, 0)['passed'])

    def test_last_edit_needs_a_new_successful_check(self):
        stop = {'type': 'step_finish', 'part': {'reason': 'stop'}}
        edit = self.check(output='Edit applied', tool='edit')
        self.assertFalse(completion_evidence([self.check(), edit, stop], True, 0)['passed'])
        self.assertTrue(completion_evidence([edit, self.check(), stop], True, 0)['passed'])
        self.assertFalse(completion_evidence([edit, self.check(), self.check('FAIL\n'), stop], True, 0)['passed'])
