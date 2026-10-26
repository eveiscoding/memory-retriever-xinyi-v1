import contextlib
import io
import json
import sqlite3
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from aml_retriever.api import ApiError
from aml_retriever.server import _Handler, RetrieverServer, _diagnostic


SECRET = "PRIVATE-memory-user-key-query-exception"


class TestSafeDiagnostics(unittest.TestCase):
    def handler(self, *, enabled=True, quiet=True, failure=None):
        responses = []

        def add(payload):
            if failure:
                raise failure
            return {"success": True}

        h = object.__new__(_Handler)
        h.server = SimpleNamespace(
            quiet=quiet, safe_diagnostics=enabled,
            config=SimpleNamespace(add_path="/add", search_path="/search", admin_enabled=False),
            service=SimpleNamespace(official_add=add, official_search=add),
            inflight=threading.BoundedSemaphore(1),
        )
        h.path = "/add?" + SECRET
        h._authorized = lambda: True
        h._read_json = lambda: {"messages": SECRET, "user_id": SECRET}
        h._send_json = lambda status, body, headers=None: responses.append((status, body))
        return h, responses

    def run_post(self, **kwargs):
        h, responses = self.handler(**kwargs)
        err, out = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            h.do_POST()
        return h, responses, err.getvalue(), out.getvalue()

    def record(self, text):
        self.assertNotIn(SECRET, text)
        r = json.loads(text)
        self.assertEqual(set(r), {"event", "stage", "elapsed_ms", "error_type"})
        self.assertGreaterEqual(r["elapsed_ms"], 0)
        return r

    def test_success_quiet_still_reports_metadata_only(self):
        _, responses, err, out = self.run_post()
        self.assertEqual(responses, [(200, {"success": True})])
        self.assertEqual(out, "")
        self.assertEqual(self.record(err)["error_type"], "none")

    def test_sqlite_exception_keeps_500_and_records_class_not_message(self):
        h, responses, err, out = self.run_post(failure=sqlite3.OperationalError(SECRET))
        self.assertEqual(responses, [(500, {"detail": {"reason": "internal error"}})])
        self.assertEqual(out, "")
        r = self.record(err)
        self.assertEqual((r["stage"], r["error_type"]), ("add", "OperationalError"))
        self.assertTrue(h.server.inflight.acquire(blocking=False))

    def test_api_error_retains_status(self):
        _, responses, err, _ = self.run_post(failure=ApiError(422, SECRET))
        self.assertEqual(responses[0][0], 422)
        self.assertEqual(self.record(err)["error_type"], "ApiError")

    def test_custom_exception_name_is_not_logged(self):
        failure = type(SECRET, (Exception,), {})(SECRET)
        _, _, err, _ = self.run_post(failure=failure)
        self.assertEqual(self.record(err)["error_type"], "OtherError")

    def test_default_disabled_is_silent_under_quiet(self):
        _, responses, err, out = self.run_post(enabled=False, failure=RuntimeError(SECRET))
        self.assertEqual(responses[0][0], 500)
        self.assertEqual((err, out), ("", ""))

    def test_safe_mode_suppresses_legacy_raw_url_access_logs(self):
        h, _ = self.handler(quiet=False)
        err, out = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            h.log_message("%s", SECRET)
            h._trace("POST", SECRET, time.time())
        self.assertEqual((err.getvalue(), out.getvalue()), ("", ""))

    def test_request_thread_failure_never_prints_traceback_or_address(self):
        server = SimpleNamespace(safe_diagnostics=True)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            try:
                raise ValueError(SECRET)
            except ValueError:
                RetrieverServer.handle_error(server, SECRET, SECRET)
        self.assertEqual(self.record(err.getvalue())["error_type"], "ValueError")
        self.assertNotIn("Traceback", err.getvalue())

    def test_diagnostic_sink_failure_does_not_change_response(self):
        class BrokenSink:
            def write(self, text):
                raise OSError(SECRET)

        h, responses = self.handler()
        with contextlib.redirect_stderr(BrokenSink()):
            h.do_POST()
        self.assertEqual(responses, [(200, {"success": True})])

    def test_elapsed_is_monotonic_and_stage_is_fixed_vocabulary(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), patch(
                "aml_retriever.server.time.monotonic", return_value=12.5):
            _diagnostic(SimpleNamespace(safe_diagnostics=True), SECRET, 10.0)
        r = self.record(err.getvalue())
        self.assertEqual((r["stage"], r["elapsed_ms"]), ("handler", 2500.0))


if __name__ == "__main__":
    unittest.main()
