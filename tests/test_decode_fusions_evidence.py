"""Public evidence consistency only; these tests do not execute GPU kernels."""
import csv
import json
from pathlib import Path
import statistics
import unittest


DATA = Path(__file__).resolve().parents[1] / "docs" / "benchmarks"


class EvidenceTests(unittest.TestCase):
    def test_raw_timing_aggregates(self):
        evidence = json.loads((DATA / "decode_fusions_2026-09-24.json").read_text())
        for row in evidence["timing"]:
            with self.subTest(profile=row["profile"], shape=row["shape_hwt"]):
                self.assertEqual(len(row["samples_ms"]), evidence["protocol"]["runs"])
                self.assertAlmostEqual(statistics.mean(row["samples_ms"]), row["mean_ms"])
                self.assertAlmostEqual(statistics.stdev(row["samples_ms"]), row["stdev_ms"])
                previous = row.get("paired_previous_research_candidate")
                if previous:
                    self.assertAlmostEqual(statistics.mean(previous["samples_ms"]), previous["mean_ms"])

    def test_chart_release_and_latest_values(self):
        with (DATA / "decode_comparison_2026-09-24.csv").open(newline="") as stream:
            rows = {r["Implementation"]: r for r in csv.DictReader(stream)}
        self.assertEqual(float(rows["PyOpt FP16 v0.2.0"]["Decode_seconds"]), 11.477)
        self.assertEqual(float(rows["PyOpt INT8 v0.2.0"]["Decode_seconds"]), 8.278)
        evidence = json.loads((DATA / "decode_fusions_2026-09-24.json").read_text())
        for row in evidence["timing"]:
            if row["shape_hwt"] != [768, 1344, 124]:
                continue
            name = "PyOpt " + row["profile"].split("_")[0].upper() + " latest fusions"
            self.assertAlmostEqual(float(rows[name]["Decode_seconds"]), row["mean_ms"] / 1000, places=6)


if __name__ == "__main__":
    unittest.main()
