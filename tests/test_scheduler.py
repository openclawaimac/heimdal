"""Scheduler: Work > Dream > Mirror priority and mode feature flags."""

import tempfile
import unittest

from tests.helpers import temp_config

from heimdal.core.scheduler import DREAM, MIRROR, WORK, Scheduler


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.config = temp_config(tempfile.mkdtemp())

    def test_mirror_disabled_by_default(self):
        scheduler = Scheduler(self.config)
        allowed, _ = scheduler.can_run(MIRROR)
        self.assertFalse(allowed)

    def test_dream_feature_flagged_off_by_default(self):
        scheduler = Scheduler(self.config)
        allowed, _ = scheduler.can_run(DREAM)
        self.assertFalse(allowed)

    def test_work_always_runs(self):
        scheduler = Scheduler(self.config)
        allowed, _ = scheduler.can_run(WORK)
        self.assertTrue(allowed)

    def test_work_preempts_background(self):
        scheduler = Scheduler(self.config)
        scheduler.dream_enabled = True
        scheduler.submit(DREAM, "dream-1")
        scheduler.submit(WORK, "work-1")
        mode, payload = scheduler.next_runnable()
        self.assertEqual(mode, WORK)
        self.assertEqual(payload, "work-1")


if __name__ == "__main__":
    unittest.main()
