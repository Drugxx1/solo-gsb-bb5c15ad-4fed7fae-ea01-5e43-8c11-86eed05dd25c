"""Tests for the judge ranking disagreement panel (raw-score based)."""
import os
import sys
import tempfile
import unittest

_tmp = tempfile.mkdtemp(prefix="calib_test_dg_")
os.environ["CALIB_DATA_DIR"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app import database as db  # noqa: E402
from app.main import app  # noqa: E402
from app.scoring import analyze, judge_disagreement  # noqa: E402

db.init_db()
client = TestClient(app)
SID = "test-disagreement"


def recs(data):
    # data: list of (contestant, judge, item, score)
    return [
        {"id": i + 1, "contestant": c, "judge": j, "item": it, "score": s}
        for i, (c, j, it, s) in enumerate(data)
    ]


def pair(rows, item="I1", ja="JA", jb="JB"):
    return next(p for p in rows if p["item"] == item and p["judge"] == ja and p["other_judge"] == jb)


class TestJudgeDisagreement(unittest.TestCase):
    def test_two_shared_all_concordant(self):
        # JA and JB agree on both: C1 > C2 in both.
        data = [
            ("C1", "JA", "I1", 9.0), ("C2", "JA", "I1", 7.0),
            ("C1", "JB", "I1", 8.0), ("C2", "JB", "I1", 6.5),
        ]
        res = judge_disagreement(recs(data))
        self.assertEqual(res["summary"]["pair_rows"], 1)
        p = res["pairs"][0]
        self.assertTrue(p["sufficient_sample"])
        self.assertEqual(p["shared_count"], 2)
        self.assertEqual(p["counts"]["concordant"], 1)
        self.assertEqual(p["counts"]["reversed"], 0)
        self.assertEqual(p["counts"]["one_tie"], 0)
        self.assertEqual(p["counts"]["both_tie"], 0)
        self.assertEqual(p["comparable_pairs"], 1)
        self.assertEqual(p["reversal_rate"], 0.0)
        self.assertEqual(p["reversed_pairs"], [])

    def test_reversed_classification_and_detail_scores(self):
        # JA: C1 > C2 (9 vs 7); JB: C2 > C1 (6 vs 8) -> reversed.
        data = [
            ("C1", "JA", "I1", 9.0), ("C2", "JA", "I1", 7.0),
            ("C1", "JB", "I1", 6.0), ("C2", "JB", "I1", 8.0),
        ]
        res = judge_disagreement(recs(data))
        p = res["pairs"][0]
        self.assertEqual(p["counts"]["reversed"], 1)
        self.assertEqual(p["comparable_pairs"], 1)
        self.assertEqual(p["reversal_rate"], 1.0)
        self.assertEqual(len(p["reversed_pairs"]), 1)
        rp = p["reversed_pairs"][0]
        self.assertEqual(rp["judge"], "JA")
        self.assertEqual(rp["other_judge"], "JB")
        # Normalized direction: contestant_a is the one JA prefers.
        self.assertEqual(rp["contestant_a"], "C1")
        self.assertEqual(rp["contestant_b"], "C2")
        self.assertEqual(rp["judge_a_score"], 9.0)
        self.assertEqual(rp["judge_b_score"], 7.0)
        self.assertEqual(rp["other_a_score"], 6.0)
        self.assertEqual(rp["other_b_score"], 8.0)

    def test_one_side_tie_excluded_from_comparable(self):
        # JA ties C1/C2 (8,8); JB strictly orders them (6 vs 9) -> one_tie.
        data = [
            ("C1", "JA", "I1", 8.0), ("C2", "JA", "I1", 8.0),
            ("C1", "JB", "I1", 6.0), ("C2", "JB", "I1", 9.0),
        ]
        res = judge_disagreement(recs(data))
        p = res["pairs"][0]
        self.assertEqual(p["counts"]["one_tie"], 1)
        self.assertEqual(p["counts"]["concordant"], 0)
        self.assertEqual(p["counts"]["reversed"], 0)
        # No pair where both judges are strict -> comparable count 0, no rate.
        self.assertEqual(p["comparable_pairs"], 0)
        self.assertIsNone(p["reversal_rate"])
        self.assertEqual(p["reversed_pairs"], [])

    def test_both_tie(self):
        data = [
            ("C1", "JA", "I1", 7.5), ("C2", "JA", "I1", 7.5),
            ("C1", "JB", "I1", 6.0), ("C2", "JB", "I1", 6.0),
        ]
        res = judge_disagreement(recs(data))
        p = res["pairs"][0]
        self.assertEqual(p["counts"]["both_tie"], 1)
        self.assertEqual(p["comparable_pairs"], 0)
        self.assertIsNone(p["reversal_rate"])

    def test_insufficient_sample_single_shared_contestant(self):
        # Only C1 was scored by both judges on I1 -> sample insufficient.
        data = [
            ("C1", "JA", "I1", 9.0), ("C2", "JA", "I1", 7.0),
            ("C1", "JB", "I1", 8.0), ("C3", "JB", "I1", 6.0),
        ]
        res = judge_disagreement(recs(data))
        self.assertEqual(res["summary"]["pair_rows"], 1)
        self.assertEqual(res["summary"]["rated_pairs"], 0)
        self.assertEqual(res["summary"]["insufficient_pairs"], 1)
        p = res["pairs"][0]
        self.assertFalse(p["sufficient_sample"])
        self.assertEqual(p["shared_count"], 1)
        # 样本不足：不给可比较对数与分歧率，也不展开任何选手对明细。
        self.assertIsNone(p["comparable_pairs"])
        self.assertIsNone(p["reversal_rate"])
        self.assertEqual(p["reversed_pairs"], [])
        for n in p["counts"].values():
            self.assertEqual(n, 0)
        self.assertIsNone(res["summary"]["reversal_rate"])

    def test_pair_without_shared_contestants_omitted(self):
        data = [
            ("C1", "JA", "I1", 9.0),
            ("C2", "JB", "I1", 6.0),
        ]
        res = judge_disagreement(recs(data))
        self.assertEqual(res["pairs"], [])
        self.assertEqual(res["summary"]["pair_rows"], 0)

    def test_rate_mixed_pairs_per_item_judge_pair(self):
        # Three shared contestants: JA 9,7,8; JB 6,8,7.
        # pairs: (C1,C2) reversed; (C1,C3) reversed (9>8 vs 6<7);
        #        (C2,C3) concordant (7<8 vs 8>7) -> reversed too.
        data = [
            ("C1", "JA", "I1", 9.0), ("C2", "JA", "I1", 7.0), ("C3", "JA", "I1", 8.0),
            ("C1", "JB", "I1", 6.0), ("C2", "JB", "I1", 8.0), ("C3", "JB", "I1", 7.0),
        ]
        res = judge_disagreement(recs(data))
        p = res["pairs"][0]
        self.assertEqual(p["counts"]["reversed"], 3)
        self.assertEqual(p["counts"]["concordant"], 0)
        self.assertEqual(p["comparable_pairs"], 3)
        self.assertAlmostEqual(p["reversal_rate"], 1.0)

    def test_concordant_plus_tie_rate_denominator(self):
        # 3 shared contestants; JA ties C2/C3 while JB strictly orders them.
        # (C1,C2): JA 9>7, JB 7>6 -> concordant
        # (C1,C3): JA 9>7, JB 7<8 -> reversed
        # (C2,C3): JA 7=7 tie, JB 6<8 -> one_tie
        data = [
            ("C1", "JA", "I1", 9.0), ("C2", "JA", "I1", 7.0), ("C3", "JA", "I1", 7.0),
            ("C1", "JB", "I1", 7.0), ("C2", "JB", "I1", 6.0), ("C3", "JB", "I1", 8.0),
        ]
        res = judge_disagreement(recs(data))
        p = res["pairs"][0]
        self.assertEqual(p["counts"]["concordant"], 1)
        self.assertEqual(p["counts"]["reversed"], 1)
        self.assertEqual(p["counts"]["one_tie"], 1)
        self.assertEqual(p["comparable_pairs"], 2)  # tie pair excluded
        self.assertAlmostEqual(p["reversal_rate"], 0.5)

    def test_scoped_by_item(self):
        # Reversed on I1, concordant on I2 -> two independent pair rows.
        data = [
            ("C1", "JA", "I1", 9.0), ("C2", "JA", "I1", 7.0),
            ("C1", "JB", "I1", 6.0), ("C2", "JB", "I1", 8.0),
            ("C1", "JA", "I2", 5.0), ("C2", "JA", "I2", 4.0),
            ("C1", "JB", "I2", 7.0), ("C2", "JB", "I2", 6.0),
        ]
        res = judge_disagreement(recs(data))
        self.assertEqual(res["summary"]["pair_rows"], 2)
        p1 = pair(res["pairs"], "I1")
        p2 = pair(res["pairs"], "I2")
        self.assertEqual(p1["counts"]["reversed"], 1)
        self.assertEqual(p2["counts"]["concordant"], 1)
        # Aggregate rate: 1 reversed / 2 comparable.
        self.assertEqual(res["summary"]["comparable_pairs"], 2)
        self.assertAlmostEqual(res["summary"]["reversal_rate"], 0.5)

    def test_three_judges_all_pairwise_rows(self):
        data = [
            (c, j, "I1", s)
            for j, scores in (("JA", (9.0, 7.0)), ("JB", (8.0, 6.0)), ("JC", (5.0, 6.0)))
            for c, s in zip(("C1", "C2"), scores)
        ]
        res = judge_disagreement(recs(data))
        self.assertEqual(res["summary"]["pair_rows"], 3)  # JA-JB, JA-JC, JB-JC
        names = {(p["judge"], p["other_judge"]) for p in res["pairs"]}
        self.assertEqual(names, {("JA", "JB"), ("JA", "JC"), ("JB", "JC")})

    def test_empty(self):
        res = judge_disagreement([])
        self.assertEqual(res["pairs"], [])
        self.assertEqual(res["summary"]["pair_rows"], 0)
        self.assertIsNone(res["summary"]["reversal_rate"])


class TestDisagreementInAnalysis(unittest.TestCase):
    def test_included_in_analyze_output(self):
        data = [
            ("C1", "JA", "I1", 9.0), ("C2", "JA", "I1", 7.0),
            ("C1", "JB", "I1", 6.0), ("C2", "JB", "I1", 8.0),
        ]
        a = analyze(recs(data), {}, "judge")
        self.assertIn("judge_disagreement", a)
        self.assertEqual(a["judge_disagreement"]["pairs"][0]["counts"]["reversed"], 1)

    def test_invariant_to_decisions_and_scope(self):
        data = [
            ("C1", "JA", "I1", 9.0), ("C2", "JA", "I1", 7.0),
            ("C1", "JB", "I1", 6.0), ("C2", "JB", "I1", 8.0),
        ]
        records = recs(data)
        baseline = analyze(records, {}, "judge")["judge_disagreement"]
        # A raw-score override (use every score as raw) must not change it.
        overridden = analyze(
            records, {r["id"]: (True, "裁决不影响分歧") for r in records}, "judge"
        )["judge_disagreement"]
        self.assertEqual(overridden, baseline)
        # Switching calibration scope must not change raw-score disagreement.
        scoped = analyze(records, {}, "judge_item")["judge_disagreement"]
        self.assertEqual(scoped, baseline)


CSV_BODY = (
    "选手,评委,项目,分数\n"
    "C1,JA,I1,9\nC2,JA,I1,7\nC3,JA,I1,8\n"
    "C1,JB,I1,6\nC2,JB,I1,8\nC3,JB,I1,7\n"
)


class TestDisagreementAPI(unittest.TestCase):
    def setUp(self):
        client.delete(f"/api/sessions/{SID}")

    def test_panel_in_analysis_and_refreshes_on_upload(self):
        r = client.post(
            f"/api/sessions/{SID}/upload?scope=judge",
            files={"file": ("d.csv", CSV_BODY.encode("utf-8-sig"), "text/csv")},
        )
        self.assertEqual(r.status_code, 200, r.text)
        dg = r.json()["judge_disagreement"]
        self.assertEqual(dg["summary"]["pair_rows"], 1)
        self.assertEqual(dg["pairs"][0]["counts"]["reversed"], 3)
        # Labels are shipped for the native page.
        self.assertEqual(dg["kind_labels"]["reversed"], "逆向")

    def test_panel_unaffected_by_decision_and_scope(self):
        client.post(
            f"/api/sessions/{SID}/upload?scope=judge",
            files={"file": ("d.csv", CSV_BODY.encode("utf-8-sig"), "text/csv")},
        )
        before = client.get(f"/api/sessions/{SID}/analysis").json()["judge_disagreement"]
        client.patch(
            f"/api/sessions/{SID}/scores/1/decision",
            json={"use_raw": True, "reason": "保留原分"},
        )
        decided = client.get(f"/api/sessions/{SID}/analysis").json()["judge_disagreement"]
        self.assertEqual(decided, before)
        client.post(f"/api/sessions/{SID}/scope", json={"scope": "judge_item"})
        scoped = client.get(f"/api/sessions/{SID}/analysis").json()["judge_disagreement"]
        self.assertEqual(scoped, before)

    def test_panel_refreshes_on_merge(self):
        client.post(
            f"/api/sessions/{SID}/upload?scope=judge",
            files={"file": ("d.csv", CSV_BODY.encode("utf-8-sig"), "text/csv")},
        )
        # Change JA's C2 score 7 -> 5: JA now orders C1>C3>C2 (was C1>C3>C2 tie-ish),
        # flipping one previously reversed pair (C2 vs C3) to a tie/concordant case.
        merged = (
            "选手,评委,项目,分数\n"
            "C1,JA,I1,9\nC2,JA,I1,5\nC3,JA,I1,8\n"
            "C1,JB,I1,6\nC2,JB,I1,8\nC3,JB,I1,7\n"
        )
        preview = client.post(
            f"/api/sessions/{SID}/merge/preview",
            files={"file": ("m.csv", merged.encode("utf-8-sig"), "text/csv")},
        )
        self.assertEqual(preview.status_code, 200, preview.text)
        p = preview.json()
        confirm = client.post(
            f"/api/sessions/{SID}/merge/confirm",
            json={"records": p["records"], "basis": p["basis"], "scope": "judge"},
        )
        self.assertEqual(confirm.status_code, 200, confirm.text)
        dg = confirm.json()["judge_disagreement"]
        counts = dg["pairs"][0]["counts"]
        # Pairs after change: C1-C2 reversed (9>5 vs 6<8), C1-C3 reversed (9>8 vs 6<7),
        # C2-C3: JA strict (5<8), JB strict (8>7) -> reversed still.
        # Sanity: panel exists and reflects the new raw scores (recomputed, not cached).
        self.assertEqual(counts["reversed"], 3)
        rp = dg["pairs"][0]["reversed_pairs"]
        entry = next(x for x in rp if x["contestant_b"] == "C2" or x["contestant_a"] == "C2")
        self.assertIn(5.0, (entry["judge_a_score"], entry["judge_b_score"]))


if __name__ == "__main__":
    unittest.main()
