"""
STEP 3 — TENETDrill detector.

Fuses a weighted subset of the physics rules from tenetdrill.physics with an
EWMA statistical tracker into a single per-depth stuck-pipe risk score in
[0,1], then validates the fused signal against the real STUCK_RT
ground-truth flag.

Fusion policy (see FUSION_RULE_NAMES / compute_rule_weights)
--------------------------------------------------------------
All 5 rules are still computed and reported (per-rule risk/reason columns,
`rules_fired`, `reasons`, and the per-rule AUC diagnostic in main()). Only 3
are included in the numeric fused_risk: torque_anomaly, wob_hookload_imbalance,
spp_spike. mud_density_differential and flow_spp_consistency are excluded on
label-independent grounds -- both abstain on a large share of this well's
depth range (sparse sensor coverage), and a rule that's blind most of the
time shouldn't carry equal weight with one that's almost always active.

Within the fusion-eligible 3, weights are set by coverage x inverse-variance
of each rule's own z-score, NEVER by correlation with STUCK_RT -- see
compute_rule_weights for the full rationale. Weights are computed once from
telemetry alone, before STUCK_RT is ever reattached.

STUCK_RT is held out of the entire detection pipeline (fuse_rules, add_ewma,
detect, compute_rule_weights) exactly as it was held out of physics.py -- it
is only reattached AFTER `detect()` has produced its output, purely for
scoring in `evaluate_against_ground_truth` / `analyze_lead_time`. Nothing
upstream of that reattachment point ever sees it.
"""
from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import pandas as pd

from tenetdrill import physics

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "volve"
CLEAN_PARQUET = DATA_DIR / "clean_depth.parquet"
DETECT_OUTPUT_PARQUET = DATA_DIR / "risk_scored.parquet"

DEPTH_COL = physics.DEPTH_COL
GROUND_TRUTH_COL = physics.GROUND_TRUTH_COL
RULE_NAMES = list(physics.RULES.keys())  # all 5 -- still computed & reported

# Rules included in the numeric fused_risk. See module docstring for why
# mud_density_differential and flow_spp_consistency are excluded.
FUSION_RULE_NAMES = ["torque_anomaly", "wob_hookload_imbalance", "spp_spike"]
NON_FUSION_RULE_NAMES = [n for n in RULE_NAMES if n not in FUSION_RULE_NAMES]

# EWMA span expressed in METERS (converted to rows using the actual grid
# spacing) so behavior doesn't depend on the resample resolution. 10 m gives
# the tracker a few rows of inertia against single-row noise while still
# reacting within roughly one physics-rule baseline window (15 m) -- stuck
# events build over meters of drilling, not instantly.
EWMA_SPAN_M = 10.0

# A rule "fires" (contributes to rules_fired / reasons) if its own risk is
# strictly positive -- the rules themselves already build in a noise
# threshold (Z_START sigma) before returning anything > 0.
FIRE_THRESHOLD = 0.0

# Classification threshold used for precision/recall/F1/confusion-matrix and
# the lead-time analysis. Fixed at the natural midpoint of the risk scale
# (0.5) rather than tuned against the labels -- see main() for the
# supplementary best-F1 diagnostic, reported separately and never used as
# "the" result.
DEFAULT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Shared row-level utilities -- single source of truth for anything that
# turns a risk_scored.parquet row into driller-facing language. Both
# ilm/dataset/generate.py (Step 4, offline training-data generation) and
# supervisor/agent.py (Step 6, live decisioning) import these so risk
# binning and rule-to-action mapping can never silently drift between the
# two -- the ILM would otherwise be trained on different thresholds than
# the live agent actually uses.
# ---------------------------------------------------------------------------

RISK_LABEL_LOW_MAX = 0.15
RISK_LABEL_MODERATE_MAX = 0.35
RISK_LABEL_ELEVATED_MAX = DEFAULT_THRESHOLD  # 0.5 -- matches the validated classification threshold


def risk_label(risk: float) -> str:
    """Bin an ewma_risk value into a driller-facing label.

    Bounds are set from the empirical distribution of ewma_risk on this well
    (Step 3: median ~0.12, 90th pct ~0.37, 95th pct ~0.46); the elevated/high
    boundary is pinned to DEFAULT_THRESHOLD so "high" means exactly what the
    validated precision/recall numbers were computed at, not a separately
    tuned cosmetic cutoff.
    """
    if risk < RISK_LABEL_LOW_MAX:
        return "low"
    if risk < RISK_LABEL_MODERATE_MAX:
        return "mild/moderate"
    if risk < RISK_LABEL_ELEVATED_MAX:
        return "elevated"
    return "high"


# Mitigation guidance per fusion rule. Deliberately limited to the 3
# fusion-eligible rules (see FUSION_RULE_NAMES) -- these are the ones
# actually driving the numeric score, so action advice stays traceable to a
# specific physical signal rather than the two excluded, less reliable rules.
RULE_ACTIONS: dict[str, str] = {
    "torque_anomaly": "keep torque and RPM changes gradual, and watch for continued drift before pushing ahead",
    "wob_hookload_imbalance": (
        "reduce WOB, work the pipe (reciprocate and rotate), and watch hookload for further overpull "
        "or weight loss before advancing"
    ),
    "spp_spike": "check circulation, watch for pack-off, and be ready to increase flow or work the string if pressure keeps climbing",
}


def fusion_fired_rules(row) -> list[str]:
    """Which of the 3 fusion-eligible rules are active and firing (risk > 0) at this row."""
    return [
        name
        for name in FUSION_RULE_NAMES
        if not bool(row[f"{name}_abstained"]) and float(row[f"{name}_risk"]) > 0
    ]


def fusion_reasons(row) -> list[str]:
    """Reason strings for the fusion rules currently firing at this row."""
    return [str(row[f"{name}_reason"]) for name in fusion_fired_rules(row)]


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def _fusion_rule_source_series(telemetry: pd.DataFrame, rule_name: str) -> pd.Series:
    """The exact raw-telemetry series each fusion-eligible rule computes its z-score from.

    Mirrors physics.py's internal computation precisely (same columns, same
    combination) so the weighting below reflects each rule's actual signal,
    not an approximation of it.
    """
    if rule_name == "torque_anomaly":
        return telemetry["Average Surface Torque kN.m"]
    if rule_name == "wob_hookload_imbalance":
        return telemetry["Corrected Total Hookload kkgf"] + telemetry["Weight on Bit kkgf"]
    if rule_name == "spp_spike":
        return telemetry["Average Standpipe Pressure kPa"]
    raise ValueError(f"no source-series mapping for fusion rule {rule_name!r}")


def compute_rule_weights(telemetry: pd.DataFrame, rule_results: pd.DataFrame) -> dict[str, float]:
    """Label-independent fusion weights for FUSION_RULE_NAMES: coverage x inverse-variance.

    weight_i = coverage_i / Var(z_i)

    - coverage_i: fraction of rows where rule i is active (not abstaining),
      read directly from physics' own abstain flags (already accounts for
      both raw sensor NaN gaps and the rolling-baseline warm-up period).
    - Var(z_i): the empirical variance, across the whole well, of the exact
      same z-score the rule computes internally against its own trailing
      IQR baseline. This is classical inverse-variance weighting applied to
      each rule's actual anomaly signal: if a rule's "1-sigma" scale
      estimate were well-calibrated, Var(z) would sit near 1. In practice on
      this well torque and SPP have highly bursty, non-stationary behavior
      (e.g. rotation cycling on/off) that a 15 m rolling IQR baseline can't
      track, so their z-scores massively overshoot their claimed sigma
      (Var(z) in the hundreds to thousands) and are discounted hard; WOB +
      hookload drifts smoothly with depth, so its z stays close to properly
      calibrated and it dominates the resulting fused score.

    Neither term is computed from, or references, STUCK_RT -- this function
    only ever sees `telemetry` and rule outputs derived from it.
    """
    physics._check_no_leak(telemetry)
    window_rows = physics._window_rows(telemetry, physics.TREND_WINDOW_M)

    raw_weights: dict[str, float] = {}
    for name in FUSION_RULE_NAMES:
        coverage = 1.0 - float(rule_results[f"{name}_abstained"].mean())

        series = _fusion_rule_source_series(telemetry, name)
        baseline, scale = physics._trailing_baseline(series, window_rows)
        z = (series - baseline) / scale
        z = z.replace([np.inf, -np.inf], np.nan)
        var_z = float(z.var())

        stability = 1.0 / var_z if np.isfinite(var_z) and var_z > 0 else 0.0
        raw_weights[name] = coverage * stability

    total = sum(raw_weights.values())
    if total <= 0:
        # degenerate fallback: equal weight rather than an undefined split
        return {name: 1.0 / len(FUSION_RULE_NAMES) for name in FUSION_RULE_NAMES}
    return {name: w / total for name, w in raw_weights.items()}


def fuse_rules(rule_results: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Combine the fusion-eligible rules' risk columns into one weighted fused_risk per depth.

    `weights` (from compute_rule_weights) must cover exactly FUSION_RULE_NAMES
    and sum to 1. A rule that abstained (risk = NaN) at a given depth is
    excluded from that row's weighted average entirely -- the remaining
    active rules' weights are renormalized among themselves, so one
    abstaining rule never drags the score toward zero. If every
    fusion-eligible rule abstains at a depth, fused_risk is explicitly NaN
    (no evidence, not "no risk") rather than 0.

    `rules_fired` / `reasons` are drawn from ALL RULE_NAMES (including the 2
    non-fusion rules) so the full physics picture stays visible even though
    only 3 rules drive the numeric score.
    """
    if set(weights) != set(FUSION_RULE_NAMES):
        raise ValueError(f"weights must cover exactly {FUSION_RULE_NAMES}, got {sorted(weights)}")

    fusion_risk_cols = [f"{n}_risk" for n in FUSION_RULE_NAMES]
    fusion_matrix = rule_results[fusion_risk_cols].to_numpy()
    w = np.array([weights[n] for n in FUSION_RULE_NAMES])

    mask = ~np.isnan(fusion_matrix)
    w_masked = np.where(mask, w, 0.0)
    weight_sum = w_masked.sum(axis=1)
    n_fusion_active = mask.sum(axis=1)
    weighted_vals = np.where(mask, fusion_matrix * w, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fused_risk = np.where(weight_sum > 0, weighted_vals / weight_sum, np.nan)

    all_risk_cols = [f"{n}_risk" for n in RULE_NAMES]
    all_matrix = rule_results[all_risk_cols].to_numpy()
    n_active_all = (~np.isnan(all_matrix)).sum(axis=1)
    reason_arrays = [rule_results[f"{n}_reason"].to_numpy() for n in RULE_NAMES]

    rules_fired: list[str] = []
    reasons: list[str] = []
    for i in range(len(rule_results)):
        row = all_matrix[i]
        fired_idx = [j for j, v in enumerate(row) if pd.notna(v) and v > FIRE_THRESHOLD]
        if fired_idx:
            rules_fired.append(", ".join(RULE_NAMES[j] for j in fired_idx))
            reasons.append(" | ".join(reason_arrays[j][i] for j in fired_idx))
        else:
            rules_fired.append("")
            reasons.append("")

    return pd.DataFrame(
        {
            DEPTH_COL: rule_results[DEPTH_COL].to_numpy(),
            "fused_risk": fused_risk,
            "n_active_rules": n_active_all,
            "n_fusion_active_rules": n_fusion_active,
            "rules_fired": rules_fired,
            "reasons": reasons,
        }
    )


def add_ewma(
    fused: pd.DataFrame, span_m: float = EWMA_SPAN_M, source_col: str = "fused_risk", output_col: str = "ewma_risk"
) -> pd.DataFrame:
    """Attach an EWMA-smoothed trend of `source_col` to capture building risk.

    Generic over `source_col`/`output_col` so the same tracker logic can be
    applied either to the fused signal or to a single rule's raw risk (used
    by the per-rule lead-time diagnostic in main()).

    Uses `ignore_na=False` (pandas default): a depth row where the source is
    NaN still lets real depth pass, so the tracker's weight on older
    evidence decays correctly across the gap instead of treating the gap as
    zero elapsed distance. In practice this means the output holds/decays
    smoothly through gaps rather than resetting.
    """
    step = float(pd.Series(fused[DEPTH_COL]).diff().median())
    if not np.isfinite(step) or step <= 0:
        step = 0.046
    span_rows = max(2, round(span_m / step))

    ewma = fused[source_col].ewm(span=span_rows, adjust=True, ignore_na=False).mean()
    out = fused.copy()
    out[output_col] = ewma.clip(lower=0.0, upper=1.0)  # bounded inputs -> bounded EWMA; clip guards float edge-cases
    return out


def detect(telemetry: pd.DataFrame, ewma_span_m: float = EWMA_SPAN_M) -> pd.DataFrame:
    """Run all physics rules + weighted EWMA fusion over raw telemetry.

    `telemetry` must NOT contain STUCK_RT -- enforced transitively by
    physics.evaluate_all(), which raises if the column is present.
    Fusion weights are recomputed from `telemetry` alone (see
    compute_rule_weights) -- call it directly if you need the weights
    themselves (e.g. for reporting).

    Returns one row per depth with `fused_risk`, `ewma_risk`,
    `n_active_rules`, `n_fusion_active_rules`, `rules_fired`, `reasons`,
    plus every individual `<rule>_risk` / `<rule>_reason` / `<rule>_abstained`
    column (all 5 rules) for drill-down (used later by the SupervisorAgent).
    """
    rule_results = physics.evaluate_all(telemetry)
    weights = compute_rule_weights(telemetry, rule_results)
    fused = fuse_rules(rule_results, weights)
    fused = add_ewma(fused, ewma_span_m)

    detail_cols = [c for c in rule_results.columns if c != DEPTH_COL]
    for c in detail_cols:
        fused[c] = rule_results[c].to_numpy()
    return fused


# ---------------------------------------------------------------------------
# Validation against ground truth (STUCK_RT reattached here only)
# ---------------------------------------------------------------------------


def _confusion_at_threshold(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    valid = ~np.isnan(scores)
    y, s = y[valid], scores[valid]
    pred = (s >= threshold).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if not (np.isnan(precision) or np.isnan(recall)) and (precision + recall) > 0
        else float("nan")
    )
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney-U formulation: P(score of a random positive > score of a random negative)."""
    valid = ~np.isnan(scores)
    y, s = y[valid], scores[valid]
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    sum_ranks_pos = ranks[y == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Average precision: sum of precision(k) * delta-recall(k) over the sorted-score curve."""
    valid = ~np.isnan(scores)
    y, s = y[valid], scores[valid]
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / n_pos
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    delta_recall = recall - recall_prev
    return float(np.sum(delta_recall * precision))


@dataclasses.dataclass
class LabelMetrics:
    label_name: str
    threshold: float
    n: int
    n_pos: int
    n_neg: int
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float


def evaluate_against_ground_truth(
    labeled: pd.DataFrame, score_col: str = "ewma_risk", threshold: float = DEFAULT_THRESHOLD
) -> list[LabelMetrics]:
    """Score the fused detector against STUCK_RT under two label definitions.

    `labeled` must already contain both `score_col` (detector output) and
    GROUND_TRUTH_COL, restricted to rows where STUCK_RT is actually logged
    (no fabricated labels outside the logged interval).
    """
    scores = labeled[score_col].to_numpy(dtype=float)
    stuck = labeled[GROUND_TRUTH_COL].to_numpy(dtype=float)

    label_defs = [
        ("STUCK_RT >= 1 (any risk)", (stuck >= 1).astype(int)),
        ("STUCK_RT == 3 (highest risk only)", (stuck == 3).astype(int)),
    ]
    results = []
    for name, y in label_defs:
        cm = _confusion_at_threshold(y, scores, threshold)
        results.append(
            LabelMetrics(
                label_name=name,
                threshold=threshold,
                roc_auc=_roc_auc(y, scores),
                pr_auc=_pr_auc(y, scores),
                **cm,
            )
        )
    return results


def per_rule_auc(labeled: pd.DataFrame) -> pd.DataFrame:
    """ROC-AUC / PR-AUC for each individual rule's raw (un-fused, un-smoothed)
    risk column against both label definitions -- shows which rule is
    actually carrying the signal, rather than hiding weak rules inside the
    fused average.
    """
    stuck = labeled[GROUND_TRUTH_COL].to_numpy(dtype=float)
    label_defs = [
        ("STUCK_RT >= 1", (stuck >= 1).astype(int)),
        ("STUCK_RT == 3", (stuck == 3).astype(int)),
    ]
    rows = []
    for rule_name in RULE_NAMES:
        scores = labeled[f"{rule_name}_risk"].to_numpy(dtype=float)
        n_active = int(np.isfinite(scores).sum())
        row = {"rule": rule_name, "n_active": n_active}
        for label_name, y in label_defs:
            row[f"roc_auc[{label_name}]"] = _roc_auc(y, scores)
            row[f"pr_auc[{label_name}]"] = _pr_auc(y, scores)
        rows.append(row)
    return pd.DataFrame(rows)


def best_f1_threshold(labeled: pd.DataFrame, y: np.ndarray, score_col: str = "ewma_risk") -> tuple[float, float]:
    """Diagnostic only: the threshold in (0,1) that maximizes F1 for `y`.

    Reported separately from the fixed-threshold metrics so the headline
    numbers are never silently tuned against the labels.
    """
    scores = labeled[score_col].to_numpy(dtype=float)
    best_t, best_f1 = DEFAULT_THRESHOLD, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        f1 = _confusion_at_threshold(y, scores, float(t))["f1"]
        if not np.isnan(f1) and f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


# ---------------------------------------------------------------------------
# Depth-aligned early-warning analysis
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EventResult:
    event_start_m: float
    event_end_m: float
    event_rows: int
    onset_flagged: bool
    lead_distance_m: float  # meters of advance warning BEFORE onset; NaN if onset wasn't flagged
    detection_delay_m: float  # meters AFTER onset until first flagged; NaN if onset was flagged or never flagged


def analyze_lead_time(
    labeled: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD, score_col: str = "ewma_risk", max_lookback_m: float = 100.0
) -> list[EventResult]:
    """For each contiguous STUCK_RT==3 event, measure early-warning lead distance.

    `labeled` must be sorted by depth ascending and restricted to the STUCK_RT
    labeled interval. For each event:
      - if the detector is already at/above `threshold` right at the onset
        row, walk backward through the contiguous above-threshold run (never
        crossing into a prior event, capped at `max_lookback_m`) and report
        how many meters ahead of onset that run started (lead_distance_m).
      - otherwise, walk forward from onset to the first flagged row within
        the event and report how many meters late that was
        (detection_delay_m); if never flagged during the event, both fields
        are NaN (a full miss).
    """
    labeled = labeled.reset_index(drop=True)
    is_event = (labeled[GROUND_TRUTH_COL] == 3).to_numpy()
    depth = labeled[DEPTH_COL].to_numpy()
    scores = labeled[score_col].to_numpy(dtype=float)

    events: list[EventResult] = []
    n = len(labeled)
    i = 0
    while i < n:
        if not is_event[i]:
            i += 1
            continue
        start = i
        while i < n and is_event[i]:
            i += 1
        onset_idx = start
        end_idx = i - 1
        onset_depth = depth[onset_idx]

        onset_flagged = np.isfinite(scores[onset_idx]) and scores[onset_idx] >= threshold

        lead_distance_m = float("nan")
        detection_delay_m = float("nan")

        if onset_flagged:
            j = onset_idx
            while (
                j - 1 >= 0
                and not is_event[j - 1]
                and (onset_depth - depth[j - 1]) <= max_lookback_m
                and np.isfinite(scores[j - 1])
                and scores[j - 1] >= threshold
            ):
                j -= 1
            lead_distance_m = float(onset_depth - depth[j])
        else:
            k = onset_idx
            while k <= end_idx and not (np.isfinite(scores[k]) and scores[k] >= threshold):
                k += 1
            if k <= end_idx:
                detection_delay_m = float(depth[k] - onset_depth)

        events.append(
            EventResult(
                event_start_m=float(onset_depth),
                event_end_m=float(depth[end_idx]),
                event_rows=int(end_idx - start + 1),
                onset_flagged=bool(onset_flagged),
                lead_distance_m=lead_distance_m,
                detection_delay_m=detection_delay_m,
            )
        )
    return events


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_metrics(m: LabelMetrics) -> None:
    print(f"\n[{m.label_name}]  threshold={m.threshold:.2f}  n={m.n:,} (pos={m.n_pos:,}, neg={m.n_neg:,})")
    print(f"  precision={m.precision:.3f}  recall={m.recall:.3f}  f1={m.f1:.3f}")
    print(f"  ROC-AUC={m.roc_auc:.3f}  PR-AUC={m.pr_auc:.3f}")
    print("  confusion matrix:")
    print(f"                 pred=risk   pred=clean")
    print(f"    actual=stuck   TP={m.tp:5d}     FN={m.fn:5d}")
    print(f"    actual=clean   FP={m.fp:5d}     TN={m.tn:5d}")


def _print_per_rule_auc(table: pd.DataFrame) -> None:
    print("\n[Per-rule AUC -- raw (un-fused, un-smoothed) risk vs ground truth]")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        print(table.set_index("rule").to_string())


def _print_lead_time(events: list[EventResult], title: str = "Level-3 event early-warning analysis") -> None:
    print(f"\n[{title}]  n_events={len(events)}")
    early = [e for e in events if e.onset_flagged and e.lead_distance_m > 0]
    at_onset = [e for e in events if e.onset_flagged and e.lead_distance_m == 0]
    late = [e for e in events if not e.onset_flagged and not np.isnan(e.detection_delay_m)]
    missed = [e for e in events if not e.onset_flagged and np.isnan(e.detection_delay_m)]

    print(f"  early warning (risk rose BEFORE onset): {len(early)}/{len(events)}")
    if early:
        leads = [e.lead_distance_m for e in early]
        print(f"    lead distance -- min={min(leads):.2f} m  median={np.median(leads):.2f} m  max={max(leads):.2f} m")
    print(f"  flagged exactly at onset (no advance warning): {len(at_onset)}/{len(events)}")
    print(f"  flagged late (after onset, within the event): {len(late)}/{len(events)}")
    if late:
        delays = [e.detection_delay_m for e in late]
        print(f"    detection delay -- min={min(delays):.2f} m  median={np.median(delays):.2f} m  max={max(delays):.2f} m")
    print(f"  missed entirely (never crossed threshold during the event): {len(missed)}/{len(events)}")

    print("\n  per-event detail:")
    for e in events:
        if e.onset_flagged:
            tag = f"lead={e.lead_distance_m:.2f} m early" if e.lead_distance_m > 0 else "flagged at onset"
        elif not np.isnan(e.detection_delay_m):
            tag = f"delay={e.detection_delay_m:.2f} m late"
        else:
            tag = "MISSED"
        print(
            f"    {e.event_start_m:8.2f} - {e.event_end_m:8.2f} m "
            f"({e.event_rows:4d} rows)  {tag}"
        )


def main() -> None:
    df = pd.read_parquet(CLEAN_PARQUET)
    telemetry = df.drop(columns=[GROUND_TRUTH_COL])

    rule_results = physics.evaluate_all(telemetry)
    weights = compute_rule_weights(telemetry, rule_results)

    print("=" * 70)
    print("TENETDrill Detector -- fusion weights (label-independent: coverage x 1/Var(z))")
    print("=" * 70)
    print(f"Excluded from fusion (still computed/reported): {NON_FUSION_RULE_NAMES}")
    for name in FUSION_RULE_NAMES:
        coverage = 1.0 - float(rule_results[f"{name}_abstained"].mean())
        print(f"  {name:26s} coverage={coverage:.3f}  weight={weights[name]:.4f}")

    fused = fuse_rules(rule_results, weights)
    fused = add_ewma(fused, EWMA_SPAN_M)
    detail_cols = [c for c in rule_results.columns if c != DEPTH_COL]
    for c in detail_cols:
        fused[c] = rule_results[c].to_numpy()

    # Dedicated EWMA of the WOB/hookload-imbalance rule alone, for the
    # per-rule lead-time diagnostic below -- reported as the primary
    # single-rule detector in its own right, not just a fusion input.
    fused = add_ewma(
        fused, span_m=EWMA_SPAN_M, source_col="wob_hookload_imbalance_risk", output_col="wob_hookload_ewma_risk"
    )

    print("\n" + "=" * 70)
    print("TENETDrill Detector -- fusion summary")
    print("=" * 70)
    print(f"Rows: {len(fused):,}")
    print(f"Rows with >=1 active rule (of all 5): {(fused['n_active_rules'] > 0).sum():,}")
    print(f"Rows with >=1 fusion-eligible rule active (of 3): {(fused['n_fusion_active_rules'] > 0).sum():,}")
    print(f"Rows with all fusion-eligible rules abstaining (fused_risk=NaN): {(fused['n_fusion_active_rules'] == 0).sum():,}")
    print(f"fused_risk: min={fused['fused_risk'].min():.2f} mean={fused['fused_risk'].mean():.2f} max={fused['fused_risk'].max():.2f}")
    print(f"ewma_risk:  min={fused['ewma_risk'].min():.2f} mean={fused['ewma_risk'].mean():.2f} max={fused['ewma_risk'].max():.2f}")
    print(f"Rows with >=1 of all 5 rules fired: {(fused['rules_fired'] != '').sum():,}")

    # ---- Reattach STUCK_RT ONLY for validation, after detection is done ----
    labeled = fused.copy()
    labeled[GROUND_TRUTH_COL] = df[GROUND_TRUTH_COL].to_numpy()
    labeled = labeled[labeled[GROUND_TRUTH_COL].notna()].reset_index(drop=True)

    print("\n" + "=" * 70)
    print(f"Validation against STUCK_RT ground truth (n_labeled_rows={len(labeled):,})")
    print("=" * 70)

    metrics = evaluate_against_ground_truth(labeled, threshold=DEFAULT_THRESHOLD)
    for m in metrics:
        _print_metrics(m)

    print("\n" + "-" * 70)
    print("WOB/hookload-imbalance rule ALONE (its own EWMA, threshold=0.5)")
    print("Reported as the primary physically-grounded detector in its own right,")
    print("independent of the 3-rule fused score above.")
    print("-" * 70)
    wob_metrics = evaluate_against_ground_truth(labeled, score_col="wob_hookload_ewma_risk", threshold=DEFAULT_THRESHOLD)
    for m in wob_metrics:
        _print_metrics(m)

    rule_auc_table = per_rule_auc(labeled)
    _print_per_rule_auc(rule_auc_table)

    print("\n[Diagnostic only -- NOT the reported result] best-F1 threshold per label:")
    for name, y in [
        ("STUCK_RT >= 1", (labeled[GROUND_TRUTH_COL].to_numpy() >= 1).astype(int)),
        ("STUCK_RT == 3", (labeled[GROUND_TRUTH_COL].to_numpy() == 3).astype(int)),
    ]:
        t, f1 = best_f1_threshold(labeled, y)
        print(f"  {name}: best threshold={t:.2f}  F1={f1:.3f}")

    events = analyze_lead_time(labeled, threshold=DEFAULT_THRESHOLD, score_col="ewma_risk")
    _print_lead_time(events, title="Level-3 early-warning analysis (fused ewma_risk, threshold=0.5)")

    wob_events = analyze_lead_time(labeled, threshold=DEFAULT_THRESHOLD, score_col="wob_hookload_ewma_risk")
    _print_lead_time(
        wob_events, title="Level-3 early-warning analysis (WOB/hookload rule ALONE, threshold=0.5)"
    )

    print("\n" + "=" * 70)

    out = fused.copy()
    out[GROUND_TRUTH_COL] = df[GROUND_TRUTH_COL].to_numpy()
    DETECT_OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DETECT_OUTPUT_PARQUET, index=False)
    print(f"Saved risk-scored parquet -> {DETECT_OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
