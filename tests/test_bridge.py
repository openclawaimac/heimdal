"""v0.2.8: local file bridge.

The bridge is a transport layer: external agents drop JSON jobs into an
inbox, Heimdal picks them up, dispatches to the existing adapters, and
writes a result JSON to an outbox.
"""

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest

from tests.helpers import repo_path, temp_config, write_temp_manifest

from heimdal import bridge
from heimdal.cli import main
from heimdal.storage import Storage


def _drop(inbox: str, name: str, payload) -> str:
    """Write a job file under inbox with the ready suffix and return the path."""
    path = os.path.join(inbox, name)
    with open(path, "w", encoding="utf-8") as fh:
        if isinstance(payload, str):
            fh.write(payload)
        else:
            json.dump(payload, fh)
    return path


def _hermes_job(job_id: str = "job-h1") -> dict:
    return Storage.read_json(repo_path("examples/bridge/hermes_job.example.json")) | {
        "job_id": job_id,
    }


def _openclaw_job(job_id: str = "job-o1") -> dict:
    payload = Storage.read_json(repo_path("examples/bridge/openclaw_job.example.json"))
    payload["job_id"] = job_id
    return payload


def _generic_job(job_id: str = "job-g1") -> dict:
    payload = Storage.read_json(repo_path("examples/bridge/generic_job.example.json"))
    payload["job_id"] = job_id
    return payload


class BridgeInitTests(unittest.TestCase):
    def test_ensure_dirs_creates_all_subdirectories(self):
        config = temp_config(tempfile.mkdtemp())
        paths = bridge.ensure_dirs(config)
        for sub in bridge.DIRS:
            self.assertTrue(os.path.isdir(paths[sub]), sub)
        # Idempotent: a second call must not raise.
        bridge.ensure_dirs(config)

    def test_cli_bridge_init_creates_directories(self):
        tmp = tempfile.mkdtemp()
        manifest = write_temp_manifest(tmp, tmp)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["bridge", "init", "--manifest", manifest])
        self.assertEqual(code, 0)
        for sub in bridge.DIRS:
            self.assertTrue(os.path.isdir(os.path.join(tmp, "bridge", sub)))


class BridgeProcessTests(unittest.TestCase):
    def setUp(self):
        self.config = temp_config(tempfile.mkdtemp())
        self.paths = bridge.ensure_dirs(self.config)
        self.defaults = {"backend": "offline", "model": None, "verifier": None}

    def _process(self) -> list[dict]:
        return bridge.process_cycle(self.config, self.paths, self.defaults, 16)

    def test_valid_hermes_job_processes_successfully(self):
        _drop(self.paths["inbox"], "job-h1.ready.json", _hermes_job("job-h1"))
        reports = self._process()
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["status"], "pass")
        # Outbox file appears; archive contains the original job.
        outbox = os.listdir(self.paths["outbox"])
        self.assertEqual(outbox, ["job-h1.result.json"])
        self.assertEqual(os.listdir(self.paths["archive"]), ["job-h1.ready.json"])
        # Inbox + processing are drained.
        self.assertEqual(os.listdir(self.paths["inbox"]), [])
        self.assertEqual(os.listdir(self.paths["processing"]), [])

    def test_outbox_result_uses_host_safe_refs(self):
        _drop(self.paths["inbox"], "job-h2.ready.json", _hermes_job("job-h2"))
        self._process()
        result = Storage.read_json(
            os.path.join(self.paths["outbox"], "job-h2.result.json")
        )
        for field in ("trace_pack_ref", "repro_pack_ref"):
            self.assertTrue(result[field], field)
            self.assertFalse(os.path.isabs(result[field]), field)
        # The bridge wrapper's input/output refs are also relative.
        self.assertFalse(os.path.isabs(result["bridge"]["input_ref"]))
        self.assertFalse(os.path.isabs(result["bridge"]["output_ref"]))
        # The wrapped adapter result carries a machine-readable code.
        self.assertEqual(result["result"]["code"], "OK")

    def test_valid_openclaw_job_processes_successfully(self):
        _drop(self.paths["inbox"], "job-o1.ready.json", _openclaw_job("job-o1"))
        reports = self._process()
        self.assertEqual(reports[0]["status"], "pass")
        self.assertEqual(reports[0]["adapter"], "openclaw")
        self.assertTrue(os.path.exists(
            os.path.join(self.paths["outbox"], "job-o1.result.json")
        ))

    def test_valid_generic_job_processes_successfully(self):
        _drop(self.paths["inbox"], "job-g1.ready.json", _generic_job("job-g1"))
        reports = self._process()
        self.assertEqual(reports[0]["status"], "pass")
        self.assertEqual(reports[0]["adapter"], "generic")

    def test_invalid_json_moves_to_failed_with_code(self):
        _drop(self.paths["inbox"], "bad.ready.json", "{not valid json")
        reports = self._process()
        self.assertEqual(reports[0]["status"], "fail")
        self.assertEqual(reports[0]["code"], "JOB_SCHEMA_INVALID")
        self.assertEqual(os.listdir(self.paths["outbox"]), [])
        failed = os.listdir(self.paths["failed"])
        self.assertIn("bad.ready.json", failed)
        # The error report is machine-readable.
        error_files = [n for n in failed if n.endswith(".error.json")]
        self.assertTrue(error_files)
        err = Storage.read_json(os.path.join(self.paths["failed"], error_files[0]))
        self.assertEqual(err["code"], "JOB_SCHEMA_INVALID")
        self.assertEqual(err["status"], "fail")

    def test_unknown_adapter_moves_to_failed_with_code(self):
        _drop(self.paths["inbox"], "u.ready.json",
              {"job_id": "job-u", "adapter": "mystery", "payload": {}})
        reports = self._process()
        self.assertEqual(reports[0]["status"], "fail")
        self.assertEqual(reports[0]["code"], "ADAPTER_UNSUPPORTED")
        err_files = [
            n for n in os.listdir(self.paths["failed"]) if n.endswith(".error.json")
        ]
        err = Storage.read_json(os.path.join(self.paths["failed"], err_files[0]))
        self.assertEqual(err["code"], "ADAPTER_UNSUPPORTED")
        self.assertIn("mystery", err["error"])

    def test_path_traversal_job_id_cannot_escape_bridge_dirs(self):
        job = _hermes_job("../../etc/evil")
        _drop(self.paths["inbox"], "trav.ready.json", job)
        self._process()
        # The result file lives inside outbox, named from the sanitized id.
        outbox = os.listdir(self.paths["outbox"])
        self.assertEqual(outbox, ["evil.result.json"])
        self.assertTrue(os.path.isfile(
            os.path.join(self.paths["outbox"], "evil.result.json")
        ))
        # And nothing escaped: /etc/evil.result.json must not exist.
        self.assertFalse(os.path.exists("/etc/evil.result.json"))

    def test_files_without_ready_suffix_are_ignored_when_fresh(self):
        # A plain .json that is too fresh -- not yet "ready" -- is left alone.
        _drop(self.paths["inbox"], "fresh.json", _hermes_job("job-f"))
        reports = self._process()
        self.assertEqual(reports, [])
        self.assertIn("fresh.json", os.listdir(self.paths["inbox"]))


class BridgeLoopTests(unittest.TestCase):
    def test_run_loop_exits_after_max_cycles(self):
        config = temp_config(tempfile.mkdtemp())
        paths = bridge.ensure_dirs(config)
        _drop(paths["inbox"], "job-loop.ready.json", _hermes_job("job-loop"))
        cycles = bridge.run_loop(
            config, paths, {"backend": "offline", "model": None, "verifier": None},
            poll_interval=0.01, max_jobs=4, max_cycles=1,
        )
        self.assertEqual(cycles, 1)
        self.assertTrue(os.path.exists(
            os.path.join(paths["outbox"], "job-loop.result.json")
        ))


class BridgeCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manifest = write_temp_manifest(self.tmp, self.tmp)

    def test_bridge_once_processes_inbox(self):
        # init -> drop example -> once -> result lands in outbox
        main(["bridge", "init", "--manifest", self.manifest])
        inbox = os.path.join(self.tmp, "bridge", "inbox")
        shutil.copy(
            repo_path("examples/bridge/hermes_job.example.json"),
            os.path.join(inbox, "job-cli.ready.json"),
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["bridge", "once", "--offline", "--manifest", self.manifest])
        self.assertEqual(code, 0)
        outbox = os.listdir(os.path.join(self.tmp, "bridge", "outbox"))
        self.assertEqual(len(outbox), 1)
        self.assertTrue(outbox[0].endswith(".result.json"))

    def test_bridge_status_reports_counts(self):
        main(["bridge", "init", "--manifest", self.manifest])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["bridge", "status", "--manifest", self.manifest])
        self.assertEqual(code, 0)
        for sub in bridge.DIRS:
            self.assertIn(sub, buf.getvalue())


if __name__ == "__main__":
    unittest.main()
