import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


TOOL = Path(__file__).resolve().parents[1] / "tools/analyze_glm_window_trace.py"
SPEC = importlib.util.spec_from_file_location("analyze_glm_window_trace", TOOL)
TRACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACE)


def window(window_id=1, request=1, tokens=2, attempt=0, outcome="completed", session="first"):
    common = dict(schema=1, session=session, device=0, request=request, window=window_id,
                  p0=20 + window_id, T=tokens, attempt=attempt)
    segments = []
    complete = outcome in ("completed", "scratch_retry")
    if complete:
        for index, (layer, stage, duration) in enumerate([
            (0, "attention", 1.0), (1, "fetch_ram", 2.0), (1, "helper_join", 0.5), (-1, "head", 0.5)
        ]):
            segments.append(dict(common, kind="segment", index=index, layer=layer, stage=stage, row=-1, ms=duration))
    end = dict(common, kind="window", layer_begin=0, layer_end=2, segments=len(segments),
               span_ms=4.0 if complete else 0.0, segments_ms=4.0 if complete else 0.0,
               complete=complete, outcome=outcome)
    return segments + [end]


def report(sampled=1, request=1, session="first", unsupported=0, errors=0):
    return dict(schema=1, session=session, device=0, request=request, kind="report", sampled=sampled,
                every=16, limit=8, unsupported=unsupported, errors=errors, windows_seen=max(32, sampled))


def records():
    return window() + [report()]


class WindowTraceTest(unittest.TestCase):
    def reject(self, value):
        with self.assertRaises(TRACE.TraceError):
            TRACE.analyze_records(value)

    def test_valid_layer_decomposition_and_outside_span(self):
        result = TRACE.analyze_records(records())
        self.assertTrue(result["complete"])
        group = result["groups"][0]
        self.assertEqual(group["span_ms"]["mean"], 4.0)
        self.assertEqual(group["top_layers"][0]["layer"], 1)
        self.assertEqual(group["top_layers"][0]["mean_ms_per_sampled_attempt"], 2.5)
        self.assertEqual(group["outside_layers_mean_ms"], 0.5)
        self.assertEqual(sum(group["stages_mean_ms"].values()), 4.0)
        self.assertEqual(result["requests"][0]["every"], 16)

    def test_t1_t2_t3_never_average_together(self):
        value = window(tokens=1) + window(17) + window(33, tokens=3) + [report(3)]
        result = TRACE.analyze_records(value)
        self.assertEqual([group["T"] for group in result["groups"]], [1, 2, 3])
        self.assertEqual([group["sampled_attempts"] for group in result["groups"]], [1, 1, 1])

    def test_scratch_retry_is_separate_and_unique_sample_count(self):
        value = window(outcome="scratch_retry") + window(attempt=1) + [report()]
        result = TRACE.analyze_records(value)
        self.assertTrue(result["complete"])
        self.assertEqual(result["sampled_windows"], 1)
        self.assertEqual(result["sampled_attempts"], 2)
        self.assertEqual({group["outcome"] for group in result["groups"]}, {"completed", "scratch_retry"})

    def test_unsupported_retry_is_visible_and_not_counted_as_complete(self):
        value = window(outcome="scratch_retry") + window(attempt=1, outcome="unsupported_retry") + [report()]
        result = TRACE.analyze_records(value)
        self.assertFalse(result["complete"])
        self.assertEqual(result["issues"][0]["reason"], "unsupported_retry")
        self.assertEqual(result["groups"][0]["outcome"], "scratch_retry")

    def test_trace_failure_retains_explicit_issue(self):
        value = window(outcome="trace_error") + [report(errors=1)]
        result = TRACE.analyze_records(value)
        self.assertFalse(result["complete"])
        self.assertEqual(result["groups"], [])
        self.assertEqual({item["reason"] for item in result["issues"]}, {"trace_error", "errors"})

    def test_unsupported_report_does_not_silently_pass(self):
        result = TRACE.analyze_records([report(0, unsupported=32)])
        self.assertFalse(result["complete"])
        self.assertIn("unsupported", [issue["reason"] for issue in result["issues"]])

    def test_no_samples_is_not_a_success(self):
        result = TRACE.analyze_records([report(0)])
        self.assertFalse(result["complete"])
        self.assertIn("no_sampled_windows", [issue["reason"] for issue in result["issues"]])

    def test_missing_report_or_window_end_rejected(self):
        self.reject(records()[:-1])
        self.reject(records()[:-2] + [report()])

    def test_missing_or_duplicate_segment_rejected(self):
        value = records()
        self.reject(value[1:])
        self.reject(value[:1] + value)
        value = records()
        value[1]["index"] = 20
        self.reject(value)

    def test_duplicate_end_or_final_report_rejected(self):
        value = records()
        self.reject(value[:-1] + [value[-2], value[-1]])
        self.reject(value + [value[-1]])

    def test_segment_count_and_coverage_rejected(self):
        for name, value in [("segments", 3), ("segments_ms", 4.5), ("span_ms", 4.5)]:
            bad = records()
            bad[-2][name] = value
            self.reject(bad)

    def test_missing_layer_or_bad_row_rejected(self):
        bad = records()
        bad[0]["layer"] = 1
        self.reject(bad)
        bad = records()
        bad[0]["row"] = 2
        self.reject(bad)
        bad = records()
        bad[0]["layer"] = 2
        self.reject(bad)

    def test_nonfinite_negative_and_bool_numbers_rejected(self):
        for value in [-1, float("nan"), float("inf"), True]:
            bad = records()
            bad[0]["ms"] = value
            self.reject(bad)
        bad = records()
        bad[0]["device"] = True
        self.reject(bad)

    def test_window_id_reset_requires_new_session(self):
        self.reject(records() + window(request=2) + [report(request=2)])
        result = TRACE.analyze_records(records() + window(session="second") + [report(session="second")])
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["requests"]), 2)

    def test_request_group_and_window_id_survive_report(self):
        value = records() + window(window_id=19, request=2) + [report(request=2)]
        result = TRACE.analyze_records(value)
        self.assertEqual(len(result["requests"]), 2)
        self.assertEqual(result["groups"][0]["sampled_attempts"], 2)

    def test_retry_missing_and_unexpected_rejected(self):
        self.reject(window(outcome="scratch_retry") + [report()])
        self.reject(window(attempt=1) + [report()])
        self.reject(window() + window(attempt=1) + [report()])
        bad = window(outcome="scratch_retry") + window(attempt=1) + [report()]
        bad[5]["p0"] = 500
        self.reject(bad)

    def test_changed_metadata_rejected(self):
        for name, value in [("p0", 500), ("T", 3), ("schema", 2), ("kind", "whatever")]:
            bad = records()
            bad[1][name] = value
            self.reject(bad)

    def test_report_must_match_sample_count(self):
        bad = records()
        bad[-1]["sampled"] = 2
        self.reject(bad)
        bad = records()
        bad[-1]["limit"] = 0
        self.reject(bad)

    def test_rounding_tolerance(self):
        value = records()
        value[-2]["span_ms"] += 0.000002
        self.assertTrue(TRACE.analyze_records(value)["complete"])

    def test_log_parser_noise_malformed_json_and_duplicate_fields(self):
        text = ["ordinary engine log\n"] + [TRACE.PREFIX + json.dumps(item) for item in records()]
        self.assertTrue(TRACE.analyze_records(TRACE.read_records(text))["complete"])
        for text in [[], ["ordinary log"], [TRACE.PREFIX + "{"],
                     [TRACE.PREFIX + '{"schema":1,"schema":1}'], [TRACE.PREFIX + "[]"]]:
            with self.assertRaises(TRACE.TraceError):
                TRACE.read_records(text)

    def test_cli_exit_status_and_output_file(self):
        with tempfile.TemporaryDirectory() as folder:
            source, dest = (Path(folder) / name for name in ("engine.log", "report.json"))
            source.write_text("\n".join(TRACE.PREFIX + json.dumps(item) for item in records()))
            completed = subprocess.run([sys.executable, str(TOOL), str(source), "--out", str(dest)],
                                       capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(json.loads(dest.read_text())["complete"])
            source.write_text(TRACE.PREFIX + json.dumps(report(0, unsupported=32)))
            completed = subprocess.run([sys.executable, str(TOOL), str(source)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 2)
            self.assertFalse(json.loads(completed.stdout)["complete"])
            source.write_text(TRACE.PREFIX + "{")
            completed = subprocess.run([sys.executable, str(TOOL), str(source)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 2)
            self.assertIn("rejected", completed.stderr)


if __name__ == "__main__":
    unittest.main()
