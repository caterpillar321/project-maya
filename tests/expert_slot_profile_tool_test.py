import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / 'tools/make_slot_profile.py'
spec = importlib.util.spec_from_file_location('slot_profile', TOOL)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SlotProfileToolTest(unittest.TestCase):
    def test_cpu_routes_are_optional(self):
        with tempfile.TemporaryDirectory() as d:
            trace = Path(d) / 'trace'
            trace.write_text('3 0 1 2 3 4 5 6 7 0 0 255\n19 0 1 2 3 4 5 6 7 0 0 255\n')
            hist, count = module.histogram(trace, [19], 0)
            self.assertEqual(count, 1)
            self.assertEqual(sum(hist[19].values()), 0)
            hist, _ = module.histogram(trace, [19], 0, True)
            self.assertEqual(sum(hist[19].values()), 8)

    def test_donor_layout_respects_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'base.json').write_text(json.dumps({3: 36, 19: 36, 20: 36}))
            (root/'experts').write_text('3 0 0 0 512\n19 0 0 0 256\n20 0 0 0 256\n')
            rows = []
            for t in range(60):
                for layer in (3, 19, 20):
                    ids = ' '.join(str((t * (1 if layer == 19 else 8) + i) % 80) for i in range(8))
                    rows.append(f'{layer} {ids} 0 0 0\n')
            (root/'trace').write_text(''.join(rows))
            subprocess.run([sys.executable, str(TOOL), '--baseline', str(root/'base.json'),
                            '--native-experts', str(root/'experts'), '--train', str(root/'trace'),
                            '--first-layer', '19', '--last-layer', '20', '--warmup', '0',
                            '--donor', '3', '28', '--output', str(root/'layout.txt')],
                           check=True, stdout=subprocess.DEVNULL)
            report = json.loads((root/'layout.json').read_text())
            slots = {int(k): v for k, v in report['slots'].items()}
            self.assertEqual(slots[3], 28)
            self.assertEqual(set(slots), {3, 19, 20})
            self.assertTrue(all(28 <= v <= 80 for v in slots.values()))
            used = slots[3] * 512 + (slots[19] + slots[20]) * 256
            self.assertEqual(used, report['used_bytes'])
            self.assertLessEqual(used, 36 * 512 + 72 * 256)


if __name__ == '__main__':
    unittest.main()
