"""
STEP 6 — SupervisorAgent.

Orchestrates a driller-facing decision purely from tenetdrill.detect()'s
already-validated output (data/volve/risk_scored.parquet): risk score,
which physics rules fired, and their plain-English reason strings. No ML
model, no GPU, no external API -- everything here is deterministic
Python + pandas, running on the same physics-rules/EWMA-fusion pipeline
validated in Step 3.

Why there's no fine-tuned model wired in
-------------------------------------------
The GPU deployment for the LoRA-fine-tuned ILM (Step 5) kept failing, so
this demo runs entirely on the validated detector output and physics
reason-strings instead. `supervisor_decision()` accepts an optional
`phrase_fn` hook specifically so a local Ollama model CAN be dropped in
later purely as a phrasing layer over the already-computed, fully-grounded
decision -- see `ollama_phrase_fn` below for the intended contract. It is
not implemented and not called by default; this module runs completely
without it today.

Ground truth (STUCK_RT) is deliberately never read here
-----------------------------------------------------------
supervisor_decision() must behave exactly as it would on a live well with
no ground-truth label available -- risk_scored.parquet's STUCK_RT column is
never touched by this module. (Step 7's dashboard reads STUCK_RT directly
from the parquet for its validation chart; that's a separate, explicitly
retrospective view, not something the live decision path sees.)
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Callable

import pandas as pd

from ilm.dataset.generate import WELL_NAME
from tenetdrill import detect

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "volve"
RISK_SCORED_PARQUET = DATA_DIR / "risk_scored.parquet"

# Generic guidance used to round recommended_actions out to 3-4 items when
# fewer than 3 fusion rules are firing (or none at all) -- a decision should
# never come back as a single bare line.
GENERIC_ACTIONS_LOW = [
    "Continue current drilling parameters -- no corrective action indicated by available sensors.",
    "Maintain routine hookload/torque/SPP monitoring as you drill ahead.",
]
GENERIC_ACTIONS_UNMONITORED = [
    "Treat this interval as unmonitored, not confirmed clean -- rely on driller judgment and other rig indicators.",
    "Flag this depth range for review once telemetry coverage resumes.",
    "Do not rely on TENETDrill's composite score at this depth.",
]
GENERIC_ACTIONS_PADDING = [
    "Keep watching the composite EWMA risk trend for continued escalation over the next several meters.",
    "Log current conditions for comparison against the next stand/connection.",
]


_CACHED_DF: pd.DataFrame | None = None


def load_risk_scored(force_reload: bool = False) -> pd.DataFrame:
    """Cached load of risk_scored.parquet so repeated API calls (Step 7) don't re-read disk each time."""
    global _CACHED_DF
    if _CACHED_DF is None or force_reload:
        _CACHED_DF = pd.read_parquet(RISK_SCORED_PARQUET)
    return _CACHED_DF


def _nearest_row(df: pd.DataFrame, depth: float) -> pd.Series:
    idx = (df[detect.DEPTH_COL] - depth).abs().idxmin()
    return df.loc[idx]


def _has_ewma_value(row: pd.Series) -> bool:
    """True once the EWMA tracker has ever seen a real reading (i.e. not before warm-up)."""
    return pd.notna(row["ewma_risk"])


def _has_fresh_evidence(row: pd.Series) -> bool:
    """True if at least one fusion rule has live (non-abstained) data AT THIS EXACT ROW."""
    return int(row["n_fusion_active_rules"]) > 0 and pd.notna(row["fused_risk"])


def data_freshness(row: pd.Series) -> str:
    """'unmonitored': no EWMA value has ever been computed yet (e.g. before warm-up) -- there is
    no risk number to show at all.
    'stale': the EWMA trend value exists (carried forward, per detect.add_ewma's documented
    gap-decay behavior) but no fusion rule has fresh data exactly at this row -- the number is a
    real carried-forward estimate, not a fresh confirmation, and must be labeled as such rather
    than silently presented as live.
    'fresh': at least one fusion rule has live data at this exact row.
    """
    if not _has_ewma_value(row):
        return "unmonitored"
    if not _has_fresh_evidence(row):
        return "stale"
    return "fresh"


def compose_explanation(row: pd.Series) -> str:
    """Plain-language risk explanation, templated from Step 2's own reason-strings."""
    freshness = data_freshness(row)

    if freshness == "unmonitored":
        return (
            "No strong physical anomaly assessment is possible here -- all three physics rules behind "
            "the composite score (torque, WOB/hookload, standpipe pressure) are missing data at this "
            "depth. Treat this depth as unmonitored, not clean."
        )

    risk = float(row["ewma_risk"])
    label = detect.risk_label(risk)

    if freshness == "stale":
        return (
            f"Stuck-pipe risk is {label} based on the trailing trend (composite score {risk:.2f}), but no "
            "fusion rule (torque, WOB/hookload, standpipe pressure) has fresh data exactly at this depth. "
            "Treat this as a carried-forward estimate from nearby readings, not a live confirmation."
        )

    reasons = detect.fusion_reasons(row)
    if label == "low":
        text = f"Stuck-pipe risk is low (composite trend score {risk:.2f})."
        if reasons:
            text += " Minor signal noted: " + " ".join(reasons)
        else:
            text += (
                " No strong physical anomaly detected in available sensors (torque, WOB/hookload, and "
                "standpipe pressure are all within normal trailing-baseline range)."
            )
        return text

    text = f"Stuck-pipe risk is {label} (composite trend score {risk:.2f})."
    if reasons:
        text += " " + " ".join(reasons)
    else:
        text += (
            " The composite trend is elevated from recent history even though no single fusion rule is "
            "firing hard exactly at this depth -- likely carryover from a recent nearby event."
        )
    return text


def compose_actions(row: pd.Series) -> list[str]:
    """3-4 recommended actions, derived from which fusion rules fired (padded with generic guidance)."""
    freshness = data_freshness(row)

    if freshness == "unmonitored":
        return GENERIC_ACTIONS_UNMONITORED[:4]

    if freshness == "stale":
        return [
            "Treat the current composite score as a carried-forward trend estimate, not a fresh reading.",
            "Cross-check with driller judgment and other rig indicators until fresh sensor data resumes.",
            "Re-evaluate as soon as torque, WOB/hookload, or standpipe pressure data resumes at this depth.",
        ]

    fired = detect.fusion_fired_rules(row)
    actions = [detect.RULE_ACTIONS[name] for name in fired] if fired else list(GENERIC_ACTIONS_LOW)

    i = 0
    while len(actions) < 3:
        actions.append(GENERIC_ACTIONS_PADDING[i % len(GENERIC_ACTIONS_PADDING)])
        i += 1
    return actions[:4]


def ollama_phrase_fn(decision: dict) -> dict:
    """Sketch of the future optional phrasing layer -- NOT used by default.

    Once a fine-tuned local Ollama model is actually deployed (via
    ilm/training/train.py --export-gguf, then `ollama create`), a function
    matching this signature could be passed as `phrase_fn=` to
    supervisor_decision(). The contract: take the already-computed,
    fully-grounded `decision` dict (risk_score, risk_level, which rules
    fired, primary_driver are all FINAL) and only restyle
    `explanation`/`recommended_actions` into more natural language --
    never invent a different risk number or a rule that didn't fire. That
    keeps the model a phrasing layer, not a decision-maker.

    Not implemented: this deployment runs GPU-free, so this function is a
    documented extension point only.
    """
    raise NotImplementedError(
        "No fine-tuned Ollama model is deployed in this environment. supervisor_decision() runs fully "
        "offline on physics templates without this hook -- see the docstring for the intended contract "
        "if one is wired in later."
    )


def supervisor_decision(
    depth: float,
    df: pd.DataFrame | None = None,
    phrase_fn: Callable[[dict], dict] | None = None,
) -> dict:
    """The single entry point Step 7's API calls directly.

    Returns a JSON-serializable dict: risk score + level, which rules fired
    (fusion-eligible and secondary), a plain-language explanation, and 3-4
    recommended actions. Never reads STUCK_RT -- see module docstring.
    """
    if df is None:
        df = load_risk_scored()
    row = _nearest_row(df, depth)

    freshness = data_freshness(row)
    ewma_risk = None if pd.isna(row["ewma_risk"]) else float(row["ewma_risk"])
    fused_risk = None if pd.isna(row["fused_risk"]) else float(row["fused_risk"])
    risk_level = "unmonitored" if freshness == "unmonitored" else detect.risk_label(ewma_risk)

    fusion_fired = detect.fusion_fired_rules(row)
    all_fired = [n for n in str(row["rules_fired"]).split(", ") if n] if row["rules_fired"] else []
    secondary_fired = [n for n in all_fired if n not in detect.FUSION_RULE_NAMES]

    primary_driver = None
    if fusion_fired:
        primary_driver = max(fusion_fired, key=lambda n: float(row[f"{n}_risk"]))

    decision = {
        "well": WELL_NAME,
        "requested_depth_m": float(depth),
        "depth_m": float(row[detect.DEPTH_COL]),
        "risk_score": ewma_risk,
        "instantaneous_risk_score": fused_risk,
        "risk_level": risk_level,
        "data_freshness": freshness,
        "sensor_coverage": {
            "fusion_rules_active": int(row["n_fusion_active_rules"]),
            "fusion_rules_total": len(detect.FUSION_RULE_NAMES),
            "all_rules_active": int(row["n_active_rules"]),
            "all_rules_total": len(detect.RULE_NAMES),
        },
        "primary_driver": primary_driver,
        "fusion_rules_fired": fusion_fired,
        "secondary_signals_fired": secondary_fired,
        "explanation": compose_explanation(row),
        "recommended_actions": compose_actions(row),
        "phrasing_backend": "template",
        "generated_by": (
            "TENETDrill physics rules + coverage/inverse-variance-weighted EWMA fusion (Steps 2-3), "
            "no ML model"
        ),
    }

    if phrase_fn is not None:
        decision = phrase_fn(decision)

    return decision


def print_decision(decision: dict) -> None:
    print("=" * 70)
    print(f"TENETDrill Supervisor Decision -- {decision['well']}")
    print("=" * 70)
    print(f"Requested depth: {decision['requested_depth_m']:.2f} m  (nearest logged: {decision['depth_m']:.2f} m)")
    print(f"Risk level: {decision['risk_level'].upper()}  (data freshness: {decision['data_freshness']})")

    rs = decision["risk_score"]
    print(f"Composite risk score (EWMA): {rs:.3f}" if rs is not None else "Composite risk score (EWMA): n/a (unmonitored)")
    irs = decision["instantaneous_risk_score"]
    print(f"Instantaneous fused risk:    {irs:.3f}" if irs is not None else "Instantaneous fused risk:    n/a")

    cov = decision["sensor_coverage"]
    print(
        f"Sensor coverage: {cov['fusion_rules_active']}/{cov['fusion_rules_total']} fusion rules active, "
        f"{cov['all_rules_active']}/{cov['all_rules_total']} rules total"
    )
    print(f"Primary driver: {decision['primary_driver'] or 'none'}")
    if decision["secondary_signals_fired"]:
        print(f"Secondary signals also firing (not part of composite score): {', '.join(decision['secondary_signals_fired'])}")

    print("\nExplanation:")
    print(f"  {decision['explanation']}")

    print("\nRecommended actions:")
    for i, a in enumerate(decision["recommended_actions"], 1):
        print(f"  {i}. {a}")

    print(f"\nGenerated by: {decision['generated_by']}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="TENETDrill SupervisorAgent CLI")
    parser.add_argument("--depth", type=float, required=True, help="measured depth in meters")
    args = parser.parse_args()

    decision = supervisor_decision(args.depth)
    print_decision(decision)


if __name__ == "__main__":
    main()
