"""
STEP 2 — Physics/rules validator for TENETDrill.

Deterministic engineering checks on raw drilling telemetry. Each rule is a
hard, bounded constraint that inspects sensor readings and local depth-trend
baselines for stuck-pipe precursor signatures (pack-off, drag, restricted
flow, influx/losses/poor hole cleaning).

CRITICAL / ground-truth hold-out
---------------------------------
STUCK_RT (the labeled stuck-pipe flag) must NEVER be read by any rule in
this module. It is held out entirely so Step 3 can validate the detector
against it -- if it leaked into the risk calculation, that validation would
be circular. `_check_no_leak` is called at the top of every public function
and RAISES if the column is present, so the constraint is enforced at
runtime, not just documented.

Every rule is a pure function: DataFrame in (depth + raw telemetry columns
only) -> DataFrame out (risk, reason, abstained), which keeps each one
independently unit-testable in isolation -- see test_physics.py.
"""
from __future__ import annotations

import dataclasses
import pathlib
from typing import Callable

import numpy as np
import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "volve"
CLEAN_PARQUET = DATA_DIR / "clean_depth.parquet"

DEPTH_COL = "Measured Depth m"
GROUND_TRUTH_COL = "STUCK_RT unitless"

# Trend-baseline rules default to a 15 m trailing window: long enough to
# represent the slow, expected depth trend (torque/SPP naturally rise with
# depth), short enough that a genuine pack-off/drag event still stands out.
TREND_WINDOW_M = 15.0
# Risk ramps linearly from 0 at Z_START sigma (ignore ordinary noise) to 1.0
# at Z_FULL sigma (strong, unambiguous anomaly) -- standard practice for
# statistical process control style anomaly bands.
Z_START = 1.0
Z_FULL = 4.0


@dataclasses.dataclass(frozen=True)
class RuleResult:
    """Output of one physics rule at one depth: bounded risk + plain reason."""

    risk: float | None  # in [0, 1]; None means the rule abstained (missing data)
    reason: str
    abstained: bool = False

    def __post_init__(self) -> None:
        if self.risk is not None and not (0.0 <= self.risk <= 1.0):
            raise ValueError(f"risk out of bounds [0,1]: {self.risk}")
        if self.abstained and self.risk is not None:
            raise ValueError("abstained rule must not carry a numeric risk")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _check_no_leak(df: pd.DataFrame) -> None:
    if GROUND_TRUTH_COL in df.columns:
        raise ValueError(
            f"Physics rules must not receive {GROUND_TRUTH_COL!r}. It is held "
            "out as ground truth for Step 3 detector validation -- drop it "
            "before calling any tenetdrill.physics function."
        )


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _window_rows(df: pd.DataFrame, window_m: float) -> int:
    step = float(df[DEPTH_COL].diff().median())
    if not np.isfinite(step) or step <= 0:
        step = 0.046  # fallback: Step-1 resample grid spacing
    return max(5, int(round(window_m / step)))


def _trailing_baseline(series: pd.Series, window_rows: int) -> tuple[pd.Series, pd.Series]:
    """Causal (past-only) rolling median + IQR-based robust scale.

    Uses `.shift(1)` so the baseline for row i never includes row i itself --
    a real-time driller-facing system can only compare "now" to "history".
    IQR/1.349 approximates a standard deviation but is far less sensitive to
    the sharp spikes we are specifically trying to detect, unlike a rolling
    mean/std would be.
    """
    trailing = series.shift(1)
    roll = trailing.rolling(window_rows, min_periods=max(5, window_rows // 4))
    median = roll.median()
    q75 = roll.quantile(0.75)
    q25 = roll.quantile(0.25)
    scale = (q75 - q25) / 1.349
    scale = scale.where(scale > 1e-6)  # avoid div-by-~0 -> NaN (abstain) instead of a blowup
    return median, scale


def _z_to_risk(z: pd.Series, z_start: float = Z_START, z_full: float = Z_FULL, two_sided: bool = False) -> pd.Series:
    zz = z.abs() if two_sided else z
    risk = (zz - z_start) / (z_full - z_start)
    return risk.clip(lower=0.0, upper=1.0)


def _finalize(risk: pd.Series, reasons: list[str], abstained: list[bool], index: pd.Index) -> pd.DataFrame:
    out = pd.DataFrame({"risk": risk.to_numpy(), "reason": reasons, "abstained": abstained}, index=index)
    bad = out["risk"].dropna()
    if len(bad) and not ((bad >= 0) & (bad <= 1)).all():
        raise ValueError("rule produced a risk value outside [0, 1] -- physically impossible output")
    return out


# ---------------------------------------------------------------------------
# Rule 1: torque anomaly vs rolling depth-trend baseline
# ---------------------------------------------------------------------------


def rule_torque_anomaly(df: pd.DataFrame, window_m: float = TREND_WINDOW_M) -> pd.DataFrame:
    """Sudden torque deviation (either direction) from the local trailing baseline.

    A RISE is the classic pack-off/stuck-pipe precursor signature. A sharp
    DROP is also flagged: on wells where the driller reduces RPM in response
    to felt drag, torque collapses mechanically along with rotation, so a
    sudden fall is itself a departure from steady-state rotary drilling and
    a companion signature worth surfacing, not just noise to ignore.
    """
    _check_no_leak(df)
    _require_columns(df, [DEPTH_COL, "Average Surface Torque kN.m"])
    torque = df["Average Surface Torque kN.m"]

    baseline, scale = _trailing_baseline(torque, _window_rows(df, window_m))
    z = (torque - baseline) / scale
    risk = _z_to_risk(z, two_sided=True)

    t = torque.to_numpy()
    b = baseline.to_numpy()
    zz = z.to_numpy()
    r = risk.to_numpy()
    reasons, abstained = [], []
    for val, base, zval, rval in zip(t, b, zz, r):
        if not (np.isfinite(val) and np.isfinite(base) and np.isfinite(zval) and np.isfinite(rval)):
            reasons.append(f"Insufficient torque history to establish a {window_m:.0f} m trailing baseline.")
            abstained.append(True)
        elif rval <= 0:
            reasons.append(
                f"Torque {val:.2f} kN·m is within the normal {window_m:.0f} m trailing "
                f"range (baseline {base:.2f} kN·m, Z={zval:.1f})."
            )
            abstained.append(False)
        elif zval > 0:
            reasons.append(
                f"Torque {val:.2f} kN·m is {zval:.1f}σ above the {window_m:.0f} m trailing "
                f"baseline ({base:.2f} kN·m) -- a rising-torque signature consistent with "
                "pack-off or a stuck-pipe precursor."
            )
            abstained.append(False)
        else:
            reasons.append(
                f"Torque {val:.2f} kN·m is {abs(zval):.1f}σ below the {window_m:.0f} m trailing "
                f"baseline ({base:.2f} kN·m) -- an abrupt torque collapse, consistent with a "
                "sudden rotation cutback (e.g. reduced RPM in response to felt drag)."
            )
            abstained.append(False)
    risk = pd.Series(np.where(np.array(abstained), np.nan, r), index=df.index)
    return _finalize(risk, reasons, abstained, df.index)


# ---------------------------------------------------------------------------
# Rule 2: WOB / hookload imbalance (drag)
# ---------------------------------------------------------------------------


def rule_wob_hookload_imbalance(df: pd.DataFrame, window_m: float = TREND_WINDOW_M) -> pd.DataFrame:
    """Deviation of (hookload + WOB) from its trailing baseline -- drag/restricted movement.

    While drilling, weight applied at the bit (WOB) is transferred off the
    hook, so `hookload + WOB` approximates the string's free-hanging weight,
    which should only drift slowly with depth. A sharp rise means more
    hookload was needed than usual to deliver the same WOB (overpull from
    friction/drag). A sharp drop means hookload fell despite WOB being held
    (weight loss -- string being gripped by the wellbore). Both are drag
    signatures, so this rule is two-sided.
    """
    _check_no_leak(df)
    _require_columns(df, [DEPTH_COL, "Weight on Bit kkgf", "Corrected Total Hookload kkgf"])
    wob = df["Weight on Bit kkgf"]
    hookload = df["Corrected Total Hookload kkgf"]
    combined = hookload + wob

    baseline, scale = _trailing_baseline(combined, _window_rows(df, window_m))
    z = (combined - baseline) / scale
    risk = _z_to_risk(z, two_sided=True)

    c = combined.to_numpy()
    w = wob.to_numpy()
    h = hookload.to_numpy()
    b = baseline.to_numpy()
    zz = z.to_numpy()
    r = risk.to_numpy()
    reasons, abstained = [], []
    for cval, wval, hval, base, zval, rval in zip(c, w, h, b, zz, r):
        if not (np.isfinite(cval) and np.isfinite(base) and np.isfinite(zval) and np.isfinite(rval)):
            reasons.append(f"Insufficient WOB/hookload history to establish a {window_m:.0f} m trailing baseline.")
            abstained.append(True)
        elif rval <= 0:
            reasons.append(
                f"Hookload {hval:.1f} kkgf and WOB {wval:.1f} kkgf are consistent with the "
                f"{window_m:.0f} m trailing baseline (combined {cval:.1f} kkgf, Z={zval:.1f})."
            )
            abstained.append(False)
        elif zval > 0:
            reasons.append(
                f"Combined hookload+WOB {cval:.1f} kkgf is {zval:.1f}σ above baseline "
                f"({base:.1f} kkgf) -- overpull consistent with drag/restricted pipe movement."
            )
            abstained.append(False)
        else:
            reasons.append(
                f"Combined hookload+WOB {cval:.1f} kkgf is {abs(zval):.1f}σ below baseline "
                f"({base:.1f} kkgf) -- weight loss consistent with the string being gripped "
                "by the wellbore (differential-sticking drag signature)."
            )
            abstained.append(False)
    risk = pd.Series(np.where(np.array(abstained), np.nan, r), index=df.index)
    return _finalize(risk, reasons, abstained, df.index)


# ---------------------------------------------------------------------------
# Rule 3: standpipe pressure spike vs baseline
# ---------------------------------------------------------------------------


def rule_spp_spike(df: pd.DataFrame, window_m: float = TREND_WINDOW_M) -> pd.DataFrame:
    """Sudden standpipe pressure deviation (either direction) from the local trailing baseline.

    A RISE is the classic pack-off/restriction signature (resistance to
    flow increasing downhole). A sharp DROP -- e.g. from a washout, a
    partial loss of returns, or a pump-rate change made to work a stuck
    string free -- is also a departure from steady-state circulation and is
    flagged, not treated as safe by default.
    """
    _check_no_leak(df)
    _require_columns(df, [DEPTH_COL, "Average Standpipe Pressure kPa"])
    spp = df["Average Standpipe Pressure kPa"]

    baseline, scale = _trailing_baseline(spp, _window_rows(df, window_m))
    z = (spp - baseline) / scale
    risk = _z_to_risk(z, two_sided=True)

    s = spp.to_numpy()
    b = baseline.to_numpy()
    zz = z.to_numpy()
    r = risk.to_numpy()
    reasons, abstained = [], []
    for val, base, zval, rval in zip(s, b, zz, r):
        if not (np.isfinite(val) and np.isfinite(base) and np.isfinite(zval) and np.isfinite(rval)):
            reasons.append(f"Insufficient SPP history to establish a {window_m:.0f} m trailing baseline.")
            abstained.append(True)
        elif rval <= 0:
            reasons.append(
                f"Standpipe pressure {val:.0f} kPa is within the normal {window_m:.0f} m "
                f"trailing range (baseline {base:.0f} kPa, Z={zval:.1f})."
            )
            abstained.append(False)
        elif zval > 0:
            reasons.append(
                f"Standpipe pressure {val:.0f} kPa is {zval:.1f}σ above the {window_m:.0f} m "
                f"trailing baseline ({base:.0f} kPa) -- a pressure-spike signature consistent "
                "with pack-off/restriction forming downhole."
            )
            abstained.append(False)
        else:
            reasons.append(
                f"Standpipe pressure {val:.0f} kPa is {abs(zval):.1f}σ below the {window_m:.0f} m "
                f"trailing baseline ({base:.0f} kPa) -- an abrupt pressure drop, consistent with "
                "a washout, loss of returns, or a sudden circulation change."
            )
            abstained.append(False)
    risk = pd.Series(np.where(np.array(abstained), np.nan, r), index=df.index)
    return _finalize(risk, reasons, abstained, df.index)


# ---------------------------------------------------------------------------
# Rule 4: mud density in/out differential
# ---------------------------------------------------------------------------

# Normal band for (density_out - density_in), g/cm3. A small positive
# differential is expected from cuttings loading in the annulus; wider or
# negative differentials indicate poor hole cleaning, influx, or dilution.
MUD_DIFF_NORMAL = (-0.02, 0.05)
MUD_DIFF_FULL_RISK = 0.15  # |excess beyond band| at which risk saturates to 1.0


def rule_mud_density_differential(df: pd.DataFrame) -> pd.DataFrame:
    """Mud density out vs in outside the expected cuttings-loading band."""
    _check_no_leak(df)
    _require_columns(df, [DEPTH_COL, "Mud Density In g/cm3", "Mud Density Out g/cm3"])
    d_in = df["Mud Density In g/cm3"]
    d_out = df["Mud Density Out g/cm3"]
    diff = d_out - d_in
    lo, hi = MUD_DIFF_NORMAL

    excess = np.where(diff > hi, diff - hi, np.where(diff < lo, lo - diff, 0.0))
    risk = np.clip(excess / MUD_DIFF_FULL_RISK, 0.0, 1.0)

    din_a, dout_a, diff_a, risk_a = d_in.to_numpy(), d_out.to_numpy(), diff.to_numpy(), risk
    reasons, abstained = [], []
    for din, dout, dv, rv in zip(din_a, dout_a, diff_a, risk_a):
        if not (np.isfinite(din) and np.isfinite(dout)):
            reasons.append("Mud density in/out not both logged at this depth.")
            abstained.append(True)
        elif rv <= 0:
            reasons.append(
                f"Mud density out ({dout:.3f} g/cm3) vs in ({din:.3f} g/cm3), Δ={dv:+.3f} "
                "g/cm3, is within the expected cuttings-loading band."
            )
            abstained.append(False)
        elif dv > hi:
            reasons.append(
                f"Mud density out ({dout:.3f} g/cm3) is Δ={dv:+.3f} g/cm3 above in "
                f"({din:.3f} g/cm3) -- above the expected cuttings-loading band, indicating "
                "poor hole cleaning and elevated pack-off risk."
            )
            abstained.append(False)
        else:
            reasons.append(
                f"Mud density out ({dout:.3f} g/cm3) is Δ={dv:+.3f} g/cm3 below in "
                f"({din:.3f} g/cm3) -- below the expected band, consistent with formation "
                "influx or return dilution."
            )
            abstained.append(False)
    risk_s = pd.Series(np.where(np.array(abstained), np.nan, risk_a), index=df.index)
    return _finalize(risk_s, reasons, abstained, df.index)


# ---------------------------------------------------------------------------
# Rule 5: flow vs SPP consistency (partial blockage)
# ---------------------------------------------------------------------------

MIN_FLOW_L_MIN = 200.0  # below this, flow/SPP relationship is not meaningful (not circulating)


def rule_flow_spp_consistency(df: pd.DataFrame, window_m: float = TREND_WINDOW_M, min_flow: float = MIN_FLOW_L_MIN) -> pd.DataFrame:
    """SPP/Flow^2 hydraulic-resistance ratio rising above baseline -- partial blockage.

    For turbulent flow through a fixed hydraulics program (bit nozzles,
    annulus), standpipe pressure scales approximately with flow squared, so
    SPP/Flow^2 is roughly constant. A rise in that ratio means more pressure
    is needed to push the same flow -- rising resistance consistent with a
    partial blockage/restriction developing.
    """
    _check_no_leak(df)
    _require_columns(df, [DEPTH_COL, "Flow Pumps L/min", "Average Standpipe Pressure kPa"])
    flow = df["Flow Pumps L/min"]
    spp = df["Average Standpipe Pressure kPa"]
    ratio = (spp / flow.pow(2)).where(flow >= min_flow)

    baseline, scale = _trailing_baseline(ratio, _window_rows(df, window_m))
    z = (ratio - baseline) / scale
    risk = _z_to_risk(z, two_sided=False)

    f = flow.to_numpy()
    s = spp.to_numpy()
    b = baseline.to_numpy()
    zz = z.to_numpy()
    r = risk.to_numpy()
    reasons, abstained = [], []
    for fval, sval, base, zval, rval in zip(f, s, b, zz, r):
        if not np.isfinite(fval):
            reasons.append("Flow rate is not logged at this depth -- hydraulic-resistance ratio not computable.")
            abstained.append(True)
        elif fval < min_flow:
            reasons.append(f"Flow {fval:.0f} L/min is below the {min_flow:.0f} L/min circulating threshold -- ratio not meaningful.")
            abstained.append(True)
        elif not (np.isfinite(sval) and np.isfinite(base) and np.isfinite(zval) and np.isfinite(rval)):
            reasons.append(f"Insufficient flow/SPP history to establish a {window_m:.0f} m trailing baseline.")
            abstained.append(True)
        elif rval <= 0:
            reasons.append(
                f"SPP {sval:.0f} kPa at flow {fval:.0f} L/min matches the {window_m:.0f} m trailing "
                f"hydraulic-resistance baseline (Z={zval:.1f})."
            )
            abstained.append(False)
        else:
            reasons.append(
                f"SPP {sval:.0f} kPa at flow {fval:.0f} L/min implies a hydraulic resistance "
                f"{zval:.1f}σ above the {window_m:.0f} m trailing baseline -- consistent with "
                "a developing partial blockage/restriction."
            )
            abstained.append(False)
    risk = pd.Series(np.where(np.array(abstained), np.nan, r), index=df.index)
    return _finalize(risk, reasons, abstained, df.index)


# ---------------------------------------------------------------------------
# Registry + orchestration
# ---------------------------------------------------------------------------

RULES: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "torque_anomaly": rule_torque_anomaly,
    "wob_hookload_imbalance": rule_wob_hookload_imbalance,
    "spp_spike": rule_spp_spike,
    "mud_density_differential": rule_mud_density_differential,
    "flow_spp_consistency": rule_flow_spp_consistency,
}


def evaluate_all(df: pd.DataFrame) -> pd.DataFrame:
    """Run every physics rule over the full telemetry DataFrame.

    `df` must NOT contain STUCK_RT -- see module docstring. Returns one row
    per input row with `<rule>_risk`, `<rule>_reason`, `<rule>_abstained`
    columns for each registered rule.
    """
    _check_no_leak(df)
    out = pd.DataFrame({DEPTH_COL: df[DEPTH_COL].to_numpy()}, index=df.index)
    for name, fn in RULES.items():
        res = fn(df)
        out[f"{name}_risk"] = res["risk"]
        out[f"{name}_reason"] = res["reason"]
        out[f"{name}_abstained"] = res["abstained"]
    return out


def explain_at_depth(df: pd.DataFrame, target_depth: float) -> dict[str, RuleResult]:
    """Convenience lookup: nearest-row RuleResult per rule at a given depth.

    Recomputes all rules over `df`, which is fine for interactive/demo use
    (Step 6 SupervisorAgent) but callers evaluating many depths should call
    `evaluate_all` once themselves and index into it instead.
    """
    _check_no_leak(df)
    idx = (df[DEPTH_COL] - target_depth).abs().idxmin()
    results: dict[str, RuleResult] = {}
    for name, fn in RULES.items():
        res = fn(df)
        row = res.loc[idx]
        results[name] = RuleResult(
            risk=None if bool(row["abstained"]) else float(row["risk"]),
            reason=str(row["reason"]),
            abstained=bool(row["abstained"]),
        )
    return results


def main() -> None:
    df = pd.read_parquet(CLEAN_PARQUET)
    # Explicit hold-out: STUCK_RT never crosses into physics.py from here on.
    telemetry = df.drop(columns=[GROUND_TRUTH_COL])

    results = evaluate_all(telemetry)

    print("=" * 70)
    print("TENETDrill Physics Rules -- self-check")
    print("(STUCK_RT dropped from the input before any rule runs)")
    print("=" * 70)
    for name in RULES:
        risk_col = results[f"{name}_risk"]
        abst_col = results[f"{name}_abstained"]
        n_abstain = int(abst_col.sum())
        n_active = len(risk_col) - n_abstain
        n_flagged = int((risk_col > 0).sum())

        bad = risk_col.dropna()
        assert ((bad >= 0) & (bad <= 1)).all(), f"{name} produced out-of-bounds risk"

        print(f"\n[{name}]")
        print(f"  rows evaluated: {n_active:,}  abstained (missing data): {n_abstain:,}")
        if n_active:
            active_risk = risk_col[risk_col.notna()]
            print(f"  risk>0 rows: {n_flagged:,} ({100 * n_flagged / n_active:.1f}%)")
            print(f"  risk stats: min={active_risk.min():.2f} mean={active_risk.mean():.2f} max={active_risk.max():.2f}")
        if n_flagged:
            top_idx = risk_col.idxmax()
            depth = telemetry.loc[top_idx, DEPTH_COL]
            print(f"  highest-risk example @ {depth:.2f} m:")
            print(f"    {results.loc[top_idx, f'{name}_reason']}")

    print("\nAll rule outputs verified within [0,1] bounds (or NaN abstain).")
    print("=" * 70)


if __name__ == "__main__":
    main()
