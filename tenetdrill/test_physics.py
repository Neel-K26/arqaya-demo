"""
Standalone unit tests for tenetdrill/physics.py using synthetic data.

No pytest dependency required -- run directly:
    python -m tenetdrill.test_physics
(Also collectible by pytest, since asserts are plain asserts.)

Each rule is tested in isolation against small, hand-built DataFrames so the
behavior (bounded output, correct abstain-on-missing-data, correct direction
of the anomaly signal) can be verified without touching the real dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tenetdrill import physics


def make_depth_index(n: int, step: float = 0.046) -> np.ndarray:
    return np.round(np.arange(n) * step + 500.0, 3)


def test_torque_anomaly_flags_sudden_rise():
    n = 400
    depth = make_depth_index(n)
    torque = np.full(n, 5.0) + np.random.default_rng(0).normal(0, 0.05, n)
    torque[-1] = 9.5  # sharp rise on the last row
    df = pd.DataFrame({physics.DEPTH_COL: depth, "Average Surface Torque kN.m": torque})

    res = physics.rule_torque_anomaly(df)
    assert res["risk"].dropna().between(0, 1).all()
    assert res["risk"].iloc[-1] > 0.5, "sudden torque rise should score high risk"
    assert not res["abstained"].iloc[-1]
    assert "above" in res["reason"].iloc[-1]

    # A flat, unremarkable torque trace should score ~0 once the baseline is warm.
    flat_row = res.iloc[200]
    assert flat_row["risk"] == 0.0 or pd.isna(flat_row["risk"])


def test_torque_anomaly_flags_sudden_drop():
    # Two-sided: an abrupt torque COLLAPSE (e.g. RPM cutback) must also be
    # flagged, not just a rise -- this well's stuck events show a collapse.
    n = 400
    depth = make_depth_index(n)
    torque = np.full(n, 5.0) + np.random.default_rng(0).normal(0, 0.05, n)
    torque[-1] = 0.5  # sharp collapse on the last row
    df = pd.DataFrame({physics.DEPTH_COL: depth, "Average Surface Torque kN.m": torque})

    res = physics.rule_torque_anomaly(df)
    assert res["risk"].iloc[-1] > 0.5, "sudden torque collapse should score high risk"
    assert "collapse" in res["reason"].iloc[-1]


def test_torque_anomaly_abstains_on_missing_data():
    n = 50
    depth = make_depth_index(n)
    torque = np.full(n, np.nan)
    df = pd.DataFrame({physics.DEPTH_COL: depth, "Average Surface Torque kN.m": torque})

    res = physics.rule_torque_anomaly(df)
    assert res["abstained"].all()
    assert res["risk"].isna().all()


def test_wob_hookload_imbalance_two_sided():
    n = 400
    depth = make_depth_index(n)
    rng = np.random.default_rng(1)
    wob = np.full(n, 10.0) + rng.normal(0, 0.1, n)
    hookload = np.full(n, 120.0) + rng.normal(0, 0.2, n)
    hookload[-1] += 20.0  # overpull spike
    df = pd.DataFrame(
        {
            physics.DEPTH_COL: depth,
            "Weight on Bit kkgf": wob,
            "Corrected Total Hookload kkgf": hookload,
        }
    )
    res = physics.rule_wob_hookload_imbalance(df)
    assert res["risk"].dropna().between(0, 1).all()
    assert res["risk"].iloc[-1] > 0.5
    assert "overpull" in res["reason"].iloc[-1]

    # weight-loss direction should also trigger, with different wording
    hookload_base = np.full(n, 120.0) + rng.normal(0, 0.2, n)
    hookload2 = hookload_base.copy()
    hookload2[-1] -= 20.0  # weight-loss dip, off the *unspiked* baseline
    df2 = df.assign(**{"Corrected Total Hookload kkgf": hookload2})
    res2 = physics.rule_wob_hookload_imbalance(df2)
    assert res2["risk"].iloc[-1] > 0.5
    assert "weight loss" in res2["reason"].iloc[-1]


def test_spp_spike_both_directions():
    n = 400
    depth = make_depth_index(n)
    rng = np.random.default_rng(2)
    spp = np.full(n, 10000.0) + rng.normal(0, 20, n)
    spp[-1] = 14000.0  # spike up
    df = pd.DataFrame({physics.DEPTH_COL: depth, "Average Standpipe Pressure kPa": spp})
    res = physics.rule_spp_spike(df)
    assert res["risk"].iloc[-1] > 0.5
    assert "above" in res["reason"].iloc[-1]

    # two-sided: a sharp drop must also be flagged (e.g. washout/loss of returns)
    spp2 = spp.copy()
    spp2[-1] = 6000.0
    df2 = df.assign(**{"Average Standpipe Pressure kPa": spp2})
    res2 = physics.rule_spp_spike(df2)
    assert res2["risk"].iloc[-1] > 0.5
    assert "drop" in res2["reason"].iloc[-1]


def test_mud_density_differential_bands():
    df = pd.DataFrame(
        {
            physics.DEPTH_COL: [100.0, 101.0, 102.0, 103.0],
            "Mud Density In g/cm3": [1.20, 1.20, 1.20, np.nan],
            "Mud Density Out g/cm3": [1.22, 1.40, 1.05, 1.25],
            # row0: diff=+0.02 -> within band -> risk 0
            # row1: diff=+0.20 -> far above band -> risk 1.0 (saturated)
            # row2: diff=-0.15 -> below band -> risk > 0
            # row3: missing input -> abstain
        }
    )
    res = physics.rule_mud_density_differential(df)
    assert res["risk"].iloc[0] == 0.0
    assert res["risk"].iloc[1] >= 0.999  # saturated at the 1.0 cap (float rounding)
    assert res["risk"].iloc[2] > 0.0
    assert res["abstained"].iloc[3]
    assert pd.isna(res["risk"].iloc[3])


def test_flow_spp_consistency_abstains_when_not_circulating():
    n = 300
    depth = make_depth_index(n)
    rng = np.random.default_rng(3)
    flow = np.full(n, 2000.0) + rng.normal(0, 5, n)
    spp = 8000.0 * (flow / 2000.0) ** 2 + rng.normal(0, 10, n)
    spp[-1] *= 1.6  # resistance spike: SPP up without flow increasing
    flow[100] = 50.0  # a row that isn't circulating

    df = pd.DataFrame(
        {
            physics.DEPTH_COL: depth,
            "Flow Pumps L/min": flow,
            "Average Standpipe Pressure kPa": spp,
        }
    )
    res = physics.rule_flow_spp_consistency(df)
    assert res["abstained"].iloc[100]
    assert res["risk"].iloc[-1] > 0.5


def test_ground_truth_leak_guard_raises():
    df = pd.DataFrame(
        {
            physics.DEPTH_COL: [1.0, 2.0],
            "Average Surface Torque kN.m": [1.0, 2.0],
            physics.GROUND_TRUTH_COL: [0.0, 3.0],
        }
    )
    for fn in physics.RULES.values():
        try:
            fn(df)
        except ValueError as e:
            assert "STUCK_RT" in str(e)
        else:
            raise AssertionError(f"{fn.__name__} did not raise on leaked ground truth column")

    try:
        physics.evaluate_all(df)
    except ValueError:
        pass
    else:
        raise AssertionError("evaluate_all did not raise on leaked ground truth column")


def test_evaluate_all_bounds_on_random_data():
    n = 500
    rng = np.random.default_rng(4)
    depth = make_depth_index(n)
    df = pd.DataFrame(
        {
            physics.DEPTH_COL: depth,
            "Weight on Bit kkgf": rng.uniform(0, 20, n),
            "Average Surface Torque kN.m": rng.uniform(0, 10, n),
            "Rate of Penetration m/h": rng.uniform(0, 100, n),
            "Average Rotary Speed rpm": rng.uniform(0, 200, n),
            "Mud Density In g/cm3": rng.uniform(1.1, 1.3, n),
            "Mud Density Out g/cm3": rng.uniform(1.1, 1.3, n),
            "Average Standpipe Pressure kPa": rng.uniform(5000, 15000, n),
            "Corrected Total Hookload kkgf": rng.uniform(80, 170, n),
            "Flow Pumps L/min": rng.uniform(500, 4000, n),
            "MWD Shock Risk unitless": np.zeros(n),
        }
    )
    # inject some NaN gaps to make sure abstain paths don't crash anything
    df.loc[10:20, "Average Surface Torque kN.m"] = np.nan
    df.loc[50:55, "Mud Density In g/cm3"] = np.nan

    results = physics.evaluate_all(df)
    for name in physics.RULES:
        risk = results[f"{name}_risk"]
        assert risk.dropna().between(0, 1).all(), f"{name} out of bounds"


def run_all():
    tests = [
        test_torque_anomaly_flags_sudden_rise,
        test_torque_anomaly_flags_sudden_drop,
        test_torque_anomaly_abstains_on_missing_data,
        test_wob_hookload_imbalance_two_sided,
        test_spp_spike_both_directions,
        test_mud_density_differential_bands,
        test_flow_spp_consistency_abstains_when_not_circulating,
        test_ground_truth_leak_guard_raises,
        test_evaluate_all_bounds_on_random_data,
    ]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} physics rule tests passed.")


if __name__ == "__main__":
    run_all()
