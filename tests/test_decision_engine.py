"""Decision Engine 단위 테스트: label 및 reasons 검증."""
import unittest

from src.decision.decision_engine import decide
from src.decision.decision_types import DecisionThresholds


class TestDecisionEngine(unittest.TestCase):
    def test_trades_insufficient_data_insufficient(self):
        """trades < min_trades → 데이터 부족(보류)"""
        th = DecisionThresholds(min_trades=100)
        r = decide({"total_trades": 50, "total_return": 0.05, "win_rate": 0.5}, th)
        self.assertEqual(r.label, "데이터 부족(보류)")
        self.assertTrue(any("trades" in x for x in r.reasons))

    def test_mdd_discard(self):
        """mdd >= discard_mdd → 폐기"""
        th = DecisionThresholds(min_trades=10, discard_mdd=0.12)
        r = decide({
            "total_trades": 150,
            "total_return": -0.01,
            "max_drawdown": -0.20,
            "win_rate": 0.4,
            "sharpe": 0.1,
        }, th)
        self.assertEqual(r.label, "폐기")
        self.assertTrue(any("mdd" in x.lower() for x in r.reasons))

    def test_candidate_requirements_met(self):
        """return>0, sharpe>0, mdd<0.05, win_rate>=0.45, trades>=100 → 실전 후보"""
        th = DecisionThresholds(min_trades=100, candidate_mdd=0.05, candidate_win_rate=0.45)
        r = decide({
            "total_trades": 120,
            "total_return": 0.03,
            "max_drawdown": -0.02,
            "win_rate": 0.5,
            "sharpe": 0.5,
        }, th)
        self.assertEqual(r.label, "실전 후보")
        self.assertIn("요건 충족", r.reasons)

    def test_improve_needed_default(self):
        """그 외 → 개선 필요"""
        th = DecisionThresholds(min_trades=100)
        r = decide({
            "total_trades": 150,
            "total_return": 0.01,
            "max_drawdown": -0.03,
            "win_rate": 0.42,
            "sharpe": 0.1,
        }, th)
        self.assertEqual(r.label, "개선 필요")

    def test_improve_win_rate_low(self):
        """win_rate < improve_win_rate → 개선 필요 (강화된 reason)"""
        th = DecisionThresholds(min_trades=100, improve_win_rate=0.40)
        r = decide({
            "total_trades": 150,
            "total_return": 0.02,
            "max_drawdown": -0.02,
            "win_rate": 0.35,
            "sharpe": 0.2,
        }, th)
        self.assertEqual(r.label, "개선 필요")
        self.assertTrue(any("win_rate" in x for x in r.reasons))

    def test_discard_return_and_sharpe(self):
        """total_return <= discard_return and sharpe <= discard_sharpe → 폐기"""
        th = DecisionThresholds(min_trades=100, discard_return=-0.02, discard_sharpe=-0.2)
        r = decide({
            "total_trades": 150,
            "total_return": -0.03,
            "max_drawdown": -0.05,
            "win_rate": 0.45,
            "sharpe": -0.5,
        }, th)
        self.assertEqual(r.label, "폐기")
        self.assertTrue(any("total_return" in x or "discard" in x for x in r.reasons))


if __name__ == "__main__":
    unittest.main()
