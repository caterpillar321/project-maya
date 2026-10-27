import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools/make_net_copy_profile.py"
spec = importlib.util.spec_from_file_location("make_net_copy_profile", TOOL)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ROW = "3 0 1 2 3 4 5 6 7 2 0 1 | 8 9\n"


class CopyProfileToolTest(unittest.TestCase):
    def test_equal_trace_weight_and_mask_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b, out = (Path(folder) / n for n in ("a", "b", "profile"))
            a.write_text(ROW)
            b.write_text(ROW * 10)
            subprocess.run([sys.executable, str(TOOL), str(a), str(b), "--out", str(out)],
                           check=True, capture_output=True)
            scores = {(int(l), int(e)): float(s) for row in out.read_text().splitlines()
                      if not row.startswith("#") for l, e, s in [row.split()]}
            self.assertEqual(scores, {(3, 0): 1.0, (3, 1): 0.25})

    def test_reject_bad_routes(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "trace"
            for text in ("", "1 2\n", ROW.replace("2 0 1", "2 0 3"),
                         ROW.replace("3 0 1 2", "46 0 1 2"), ROW.replace("3 0 1 2", "3 0 0 2")):
                p.write_text(text)
                with self.assertRaises(ValueError):
                    module.trace_scores(p)


if __name__ == "__main__":
    unittest.main()
