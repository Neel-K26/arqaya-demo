"""
Standalone unit tests for tenetdrill/detect.py using synthetic data.

Run directly:
    python -m tenetdrill.test_detect

Focus areas: fusion renormalizes over active rules without diluting toward
zero when a rule abstains, EWMA persists/decays sensibly across abstain
gaps, and the hand-rolled ROC-AUC/PR-AUC implementations agree with their
textbook closed-form answers on known-answer synthetic cases (no sklearn
dependency to cross-check against, so these known-answer cases ARE the
correctness check).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tenetdrill import detect, physics


EQUAL_FUSION_WEIGHTS = {n: 1.0 / 3 for n in detect.FUSION_RULE_NAMES}


def _make_rule_results(row: dict) -> pd.DataFrame:
    """Build a 1-row rule_results DataFrame covering all 5 rules, defaults to abstain."""
    base = {detect.DEPTH_COL: [100.0]}
    for n in detect.RULE_NAMES:
        base[f"{n}_risk"] = [row.get(f"{n}_risk", np.nan)]
        base[f"{n}_reason"] = [row.get(f"{n}_reason", "abstain: no data")]
    return pd.DataFrame(base)


def test_fuse_rules_abstain_does_not_dilute():
    # Two fusion-eligible rules active with EQUAL risk, one (spp) abstains --
    # fused_risk should equal that shared value regardless of relative
    # weights among the active two, not be diluted by treating the abstain as 0.
    rule_results = _make_rule_results(
        {
            "torque_anomaly_risk": 0.8,
            "torque_anomaly_reason": "torque high",
            "wob_hookload_imbalance_risk": 0.8,
            "wob_hookload_imbalance_reason": "overpull",
        }
    )
    weights = {"torque_anomaly": 0.6, "wob_hookload_imbalance": 0.3, "spp_spike": 0.1}
    fused = detect.fuse_rules(rule_results, weights)
    assert fused["n_active_rules"].iloc[0] == 2
    assert fused["n_fusion_active_rules"].iloc[0] == 2
    assert abs(fused["fused_risk"].iloc[0] - 0.8) < 1e-9, "abstaining rule must not dilute the mean toward 0"
    assert "torque_anomaly" in fused["rules_fired"].iloc[0]
    assert "wob_hookload_imbalance" in fused["rules_fired"].iloc[0]
    assert "spp_spike" not in fused["rules_fired"].iloc[0]


def test_fuse_rules_weighted_average_respects_weights():
    # torque=0.9 (weight 0.6), wob_hookload=0.3 (weight 0.3), spp abstains
    # (weight 0.1). Renormalized over the active pair: (0.6*0.9+0.3*0.3)/0.9 = 0.7
    rule_results = _make_rule_results(
        {
            "torque_anomaly_risk": 0.9,
            "wob_hookload_imbalance_risk": 0.3,
        }
    )
    weights = {"torque_anomaly": 0.6, "wob_hookload_imbalance": 0.3, "spp_spike": 0.1}
    fused = detect.fuse_rules(rule_results, weights)
    assert abs(fused["fused_risk"].iloc[0] - 0.7) < 1e-9


def test_fuse_rules_non_fusion_rules_still_reported_but_not_fused():
    # mud_density_differential fires strongly but must NOT affect fused_risk
    # since it's excluded from FUSION_RULE_NAMES; it must still show up in
    # rules_fired/reasons for transparency.
    rule_results = _make_rule_results(
        {
            "torque_anomaly_risk": 0.4,
            "mud_density_differential_risk": 1.0,
            "mud_density_differential_reason": "big density swing",
        }
    )
    weights = {"torque_anomaly": 1.0, "wob_hookload_imbalance": 0.0, "spp_spike": 0.0}
    fused = detect.fuse_rules(rule_results, weights)
    assert abs(fused["fused_risk"].iloc[0] - 0.4) < 1e-9
    assert "mud_density_differential" in fused["rules_fired"].iloc[0]
    assert "big density swing" in fused["reasons"].iloc[0]


def test_fuse_rules_all_abstain_is_nan_not_zero():
    rule_results = _make_rule_results({})
    fused = detect.fuse_rules(rule_results, EQUAL_FUSION_WEIGHTS)
    assert fused["n_active_rules"].iloc[0] == 0
    assert fused["n_fusion_active_rules"].iloc[0] == 0
    assert pd.isna(fused["fused_risk"].iloc[0]), "all rules abstaining must yield NaN, not a false zero"
    assert fused["rules_fired"].iloc[0] == ""


def test_fuse_rules_rejects_wrong_weight_keys():
    rule_results = _make_rule_results({"torque_anomaly_risk": 0.5})
    try:
        detect.fuse_rules(rule_results, {"torque_anomaly": 1.0})  # missing 2 required keys
    except ValueError:
        pass
    else:
        raise AssertionError("fuse_rules should reject a weights dict that doesn't cover FUSION_RULE_NAMES exactly")


def test_compute_rule_weights_is_label_independent_and_normalized():
    n = 2000
    rng = np.random.default_rng(7)
    depth = np.round(np.arange(n) * 0.046, 3) + 300.0

    # Torque: mostly tight noise but with sparse large spikes (like real
    # connection/rotation-cycling artifacts) a rolling IQR baseline can't
    # characterize -> inflated Var(z). WOB+hookload stays clean Gaussian
    # noise around a large stable mean -> Var(z) near 1. Inverse-variance
    # weighting should favor the latter.
    torque = np.full(n, 5.0) + rng.normal(0, 0.3, n)
    spike_idx = rng.choice(n, size=n // 20, replace=False)
    torque[spike_idx] += rng.choice([-1, 1], size=len(spike_idx)) * rng.uniform(3, 6, len(spike_idx))
    torque = np.clip(torque, 0, 100)

    telemetry = pd.DataFrame(
        {
            detect.DEPTH_COL: depth,
            "Average Surface Torque kN.m": torque,
            "Weight on Bit kkgf": np.clip(10.0 + rng.normal(0, 0.05, n), 0, 60),
            "Corrected Total Hookload kkgf": 130.0 + rng.normal(0, 0.05, n),  # smooth -> low Var(z)
            "Average Standpipe Pressure kPa": np.clip(10000.0 + rng.normal(0, 500.0, n), 0, 60000),
            "Rate of Penetration m/h": np.clip(40.0 + rng.normal(0, 10.0, n), 0, 300),
            "Average Rotary Speed rpm": np.clip(120.0 + rng.normal(0, 10.0, n), 0, 400),
            "Mud Density In g/cm3": 1.2 + rng.normal(0, 0.01, n),
            "Mud Density Out g/cm3": 1.22 + rng.normal(0, 0.01, n),
            "Flow Pumps L/min": np.clip(2500.0 + rng.normal(0, 100.0, n), 0, 10000),
            "MWD Shock Risk unitless": np.zeros(n),
        }
    )
    # no STUCK_RT column anywhere in this test's telemetry -- weights must
    # not require or reference it (compute_rule_weights would raise if a
    # caller mistakenly passed it in, same guard as physics.py)
    rule_results = physics.evaluate_all(telemetry)
    weights = detect.compute_rule_weights(telemetry, rule_results)

    assert set(weights) == set(detect.FUSION_RULE_NAMES)
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert all(w >= 0 for w in weights.values())
    # the smooth, tightly-varying combined WOB+hookload signal should heavily
    # outweigh the noisy torque signal under inverse-variance weighting
    assert weights["wob_hookload_imbalance"] > weights["torque_anomaly"]


def test_compute_rule_weights_raises_on_leaked_ground_truth():
    telemetry = pd.DataFrame(
        {
            detect.DEPTH_COL: [1.0, 2.0],
            "Average Surface Torque kN.m": [1.0, 2.0],
            physics.GROUND_TRUTH_COL: [0.0, 3.0],
        }
    )
    try:
        detect.compute_rule_weights(telemetry, pd.DataFrame({detect.DEPTH_COL: [1.0, 2.0]}))
    except ValueError as e:
        assert "STUCK_RT" in str(e)
    else:
        raise AssertionError("compute_rule_weights did not raise on leaked ground truth column")


def test_ewma_persists_and_decays_across_abstain_gap():
    depth = np.round(np.arange(20) * 0.5, 3)
    fused_risk = np.full(20, np.nan)
    fused_risk[0:3] = 0.9  # early spike
    fused_risk[3:15] = np.nan  # long abstain gap
    fused_risk[15] = 0.1  # low reading resumes
    fused = pd.DataFrame({detect.DEPTH_COL: depth, "fused_risk": fused_risk})
    out = detect.add_ewma(fused, span_m=2.0)

    assert out["ewma_risk"].iloc[0:3].notna().all()
    # during the gap, ewma_risk should hold/decay from the last known value,
    # not jump to NaN or reset to 0
    assert out["ewma_risk"].iloc[3:15].notna().all()
    assert out["ewma_risk"].iloc[14] <= out["ewma_risk"].iloc[2] + 1e-9
    assert out["ewma_risk"].between(0, 1).all()


def test_roc_auc_known_answers():
    y = np.array([0, 0, 0, 1, 1, 1])
    perfect = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert abs(detect._roc_auc(y, perfect) - 1.0) < 1e-9

    inverted = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    assert abs(detect._roc_auc(y, inverted) - 0.0) < 1e-9

    tie_all = np.array([0.5] * 6)
    assert abs(detect._roc_auc(y, tie_all) - 0.5) < 1e-9

    with_nan = np.array([0.1, 0.2, np.nan, 0.7, 0.8, 0.9])
    y2 = np.array([0, 0, 0, 1, 1, 1])
    assert abs(detect._roc_auc(y2, with_nan) - 1.0) < 1e-9, "NaN scores should be dropped, not break the ranking"


def test_pr_auc_known_answers():
    y = np.array([0, 0, 0, 1, 1, 1])
    perfect = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert abs(detect._pr_auc(y, perfect) - 1.0) < 1e-9

    # worst-case ranking: all negatives ranked above all positives
    inverted = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    pr = detect._pr_auc(y, inverted)
    assert pr < 0.6, f"inverted ranking should score poorly on PR-AUC, got {pr}"


def test_confusion_matrix_counts():
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.4, 0.6, 0.1])  # threshold 0.5 -> pred [1,0,1,0]
    cm = detect._confusion_at_threshold(y, scores, 0.5)
    assert cm["tp"] == 1  # y=1,score=0.9
    assert cm["fn"] == 1  # y=1,score=0.4
    assert cm["fp"] == 1  # y=0,score=0.6
    assert cm["tn"] == 1  # y=0,score=0.1
    assert abs(cm["precision"] - 0.5) < 1e-9
    assert abs(cm["recall"] - 0.5) < 1e-9
    assert abs(cm["f1"] - 0.5) < 1e-9


def test_analyze_lead_time_detects_early_warning():
    # risk rises 2 rows (1.0 m) before the STUCK_RT==3 event starts, and
    # stays elevated through the whole event.
    depth = np.round(np.arange(10) * 0.5, 3)
    ewma = np.array([0.1, 0.1, 0.1, 0.6, 0.7, 0.8, 0.8, 0.2, 0.1, 0.1])
    stuck = np.array([0, 0, 0, 0, 0, 3, 3, 0, 0, 0], dtype=float)
    labeled = pd.DataFrame({detect.DEPTH_COL: depth, "ewma_risk": ewma, detect.GROUND_TRUTH_COL: stuck})

    events = detect.analyze_lead_time(labeled, threshold=0.5)
    assert len(events) == 1
    e = events[0]
    assert e.onset_flagged
    assert abs(e.lead_distance_m - 1.0) < 1e-9, f"expected 1.0 m lead, got {e.lead_distance_m}"


def test_analyze_lead_time_detects_late_flag_and_miss():
    depth = np.round(np.arange(10) * 0.5, 3)
    # event is rows 5-6; detector only crosses threshold at row 6 -> late by 0.5m
    ewma_late = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.6, 0.1, 0.1, 0.1])
    stuck = np.array([0, 0, 0, 0, 0, 3, 3, 0, 0, 0], dtype=float)
    labeled = pd.DataFrame({detect.DEPTH_COL: depth, "ewma_risk": ewma_late, detect.GROUND_TRUTH_COL: stuck})
    events = detect.analyze_lead_time(labeled, threshold=0.5)
    assert not events[0].onset_flagged
    assert abs(events[0].detection_delay_m - 0.5) < 1e-9

    # never crosses threshold during the event -> full miss
    ewma_miss = np.full(10, 0.1)
    labeled2 = pd.DataFrame({detect.DEPTH_COL: depth, "ewma_risk": ewma_miss, detect.GROUND_TRUTH_COL: stuck})
    events2 = detect.analyze_lead_time(labeled2, threshold=0.5)
    assert not events2[0].onset_flagged
    assert np.isnan(events2[0].detection_delay_m)


def run_all():
    tests = [
        test_fuse_rules_abstain_does_not_dilute,
        test_fuse_rules_weighted_average_respects_weights,
        test_fuse_rules_non_fusion_rules_still_reported_but_not_fused,
        test_fuse_rules_all_abstain_is_nan_not_zero,
        test_fuse_rules_rejects_wrong_weight_keys,
        test_compute_rule_weights_is_label_independent_and_normalized,
        test_compute_rule_weights_raises_on_leaked_ground_truth,
        test_ewma_persists_and_decays_across_abstain_gap,
        test_roc_auc_known_answers,
        test_pr_auc_known_answers,
        test_confusion_matrix_counts,
        test_analyze_lead_time_detects_early_warning,
        test_analyze_lead_time_detects_late_flag_and_miss,
    ]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} detector tests passed.")


if __name__ == "__main__":
    run_all()
