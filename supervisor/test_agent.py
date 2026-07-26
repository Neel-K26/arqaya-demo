"""
Standalone unit tests for supervisor/agent.py using synthetic rows.

Run directly:
    python -m supervisor.test_agent
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from supervisor import agent
from tenetdrill import detect


def _make_row(**overrides) -> pd.Series:
    base = {detect.DEPTH_COL: 500.0, "fused_risk": 0.0, "ewma_risk": 0.0, "n_fusion_active_rules": 3, "n_active_rules": 5, "rules_fired": ""}
    for name in detect.RULE_NAMES:
        base[f"{name}_risk"] = 0.0
        base[f"{name}_reason"] = f"{name} quiet"
        base[f"{name}_abstained"] = False
    base.update(overrides)
    return pd.Series(base)


def test_unmonitored_when_ewma_never_computed():
    # True "no data ever" case -- no EWMA value exists yet (e.g. before warm-up)
    row = _make_row(n_fusion_active_rules=0, fused_risk=np.nan, ewma_risk=np.nan)
    assert agent.data_freshness(row) == "unmonitored"
    actions = agent.compose_actions(row)
    assert 3 <= len(actions) <= 4
    assert "unmonitored" in agent.compose_explanation(row).lower()


def test_stale_when_ewma_carried_forward_but_no_fresh_evidence():
    # This is the real bug found via manual API testing: at the deepest logged
    # depth, all 3 fusion rules had abstained AT THAT ROW, but ewma_risk still
    # carried a real value forward from nearby readings (detect.add_ewma's
    # documented gap-decay behavior). risk_level must NOT say "unmonitored"
    # while risk_score is a real non-null number -- that combination is what
    # broke. It must be labeled "stale" instead, distinct from both "fresh"
    # and true "unmonitored".
    row = _make_row(n_fusion_active_rules=0, fused_risk=np.nan, ewma_risk=0.142)
    assert agent.data_freshness(row) == "stale"

    explanation = agent.compose_explanation(row)
    assert "carried-forward" in explanation.lower()
    assert "0.14" in explanation

    actions = agent.compose_actions(row)
    assert 3 <= len(actions) <= 4
    assert any("carried-forward" in a.lower() for a in actions)


def test_low_risk_no_rules_fired_still_returns_3_to_4_actions():
    row = _make_row(ewma_risk=0.05, fused_risk=0.0)
    assert agent.data_freshness(row) == "fresh"
    actions = agent.compose_actions(row)
    assert 3 <= len(actions) <= 4, f"expected 3-4 actions, got {len(actions)}"
    explanation = agent.compose_explanation(row)
    assert "low" in explanation.lower()
    assert "no strong physical anomaly" in explanation.lower()


def test_high_risk_multiple_rules_fired():
    row = _make_row(
        ewma_risk=0.65,
        fused_risk=0.9,
        rules_fired="torque_anomaly, wob_hookload_imbalance, spp_spike",
        torque_anomaly_risk=0.7,
        wob_hookload_imbalance_risk=0.95,
        spp_spike_risk=0.4,
    )
    fired = detect.fusion_fired_rules(row)
    assert set(fired) == {"torque_anomaly", "wob_hookload_imbalance", "spp_spike"}

    actions = agent.compose_actions(row)
    assert 3 <= len(actions) <= 4
    assert actions[0] == detect.RULE_ACTIONS["torque_anomaly"]

    explanation = agent.compose_explanation(row)
    assert "high" in explanation.lower()
    assert "torque anomaly quiet" not in explanation  # sanity: uses real reason text, not label


def test_primary_driver_is_highest_risk_fusion_rule():
    df = pd.DataFrame(
        [
            _make_row(
                **{detect.DEPTH_COL: 600.0},
                ewma_risk=0.55,
                fused_risk=0.8,
                rules_fired="torque_anomaly, wob_hookload_imbalance",
                torque_anomaly_risk=0.3,
                wob_hookload_imbalance_risk=0.9,
            )
        ]
    )
    decision = agent.supervisor_decision(600.0, df=df)
    assert decision["primary_driver"] == "wob_hookload_imbalance"
    assert decision["risk_level"] == "high"
    assert 3 <= len(decision["recommended_actions"]) <= 4
    assert decision["risk_score"] == 0.55


def test_supervisor_decision_never_touches_ground_truth_column():
    df = pd.DataFrame([_make_row(**{detect.DEPTH_COL: 700.0}, ewma_risk=0.1, fused_risk=0.0)])
    assert detect.GROUND_TRUTH_COL not in df.columns
    decision = agent.supervisor_decision(700.0, df=df)
    assert detect.GROUND_TRUTH_COL not in str(decision)


def test_phrase_fn_hook_applies_and_is_optional():
    df = pd.DataFrame([_make_row(**{detect.DEPTH_COL: 800.0}, ewma_risk=0.1, fused_risk=0.0)])

    def uppercase_phraser(decision: dict) -> dict:
        decision = dict(decision)
        decision["explanation"] = decision["explanation"].upper()
        return decision

    plain = agent.supervisor_decision(800.0, df=df)
    phrased = agent.supervisor_decision(800.0, df=df, phrase_fn=uppercase_phraser)
    assert phrased["explanation"] == plain["explanation"].upper()
    assert phrased["risk_score"] == plain["risk_score"]  # hook must not touch the numbers


def test_ollama_phrase_fn_is_documented_stub_not_wired_in():
    try:
        agent.ollama_phrase_fn({})
    except NotImplementedError:
        pass
    else:
        raise AssertionError("ollama_phrase_fn should be an explicit not-implemented stub today")


def test_nearest_row_picks_closest_depth():
    df = pd.DataFrame(
        [
            _make_row(**{detect.DEPTH_COL: 100.0}),
            _make_row(**{detect.DEPTH_COL: 105.0}),
            _make_row(**{detect.DEPTH_COL: 110.0}),
        ]
    )
    decision = agent.supervisor_decision(103.9, df=df)
    assert decision["depth_m"] == 105.0
    assert decision["requested_depth_m"] == 103.9


def run_all():
    tests = [
        test_unmonitored_when_ewma_never_computed,
        test_stale_when_ewma_carried_forward_but_no_fresh_evidence,
        test_low_risk_no_rules_fired_still_returns_3_to_4_actions,
        test_high_risk_multiple_rules_fired,
        test_primary_driver_is_highest_risk_fusion_rule,
        test_supervisor_decision_never_touches_ground_truth_column,
        test_phrase_fn_hook_applies_and_is_optional,
        test_ollama_phrase_fn_is_documented_stub_not_wired_in,
        test_nearest_row_picks_closest_depth,
    ]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} supervisor tests passed.")


if __name__ == "__main__":
    run_all()
