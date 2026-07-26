"""
STEP 4 — ILM training-data generator.

Deterministically generates instruction/response JSONL pairs for LoRA
fine-tuning (Step 5), grounded ONLY in tenetdrill.detect()'s actual output
for this well (fused_risk, ewma_risk, per-rule risk/reason/abstained
columns from data/volve/risk_scored.parquet) -- NEVER in STUCK_RT.

Why STUCK_RT is excluded here too
-----------------------------------
STUCK_RT is the Volve ground-truth label used to VALIDATE the detector in
Step 3. It will not exist on a live well at inference time -- a real
deployment only ever has telemetry and TENETDrill's own physics/EWMA
output. Training the ILM to reference STUCK_RT would teach it an
omniscience it will never actually have, and the model would learn to
sound confident in a way it can't back up outside this training set. The
generator drops the column immediately after loading and never looks at it
again.

Honesty by construction
-------------------------
Every rule's `_reason` text (from physics.py) is already well-formed for
BOTH the active and abstained case -- e.g. an abstained torque rule reason
reads "Insufficient torque history to establish a 15 m trailing baseline,"
not silence. Answers here are built directly from that text, so:
  - a rule that fires produces a specific, numeric, grounded claim
  - a rule that's quiet says there's no anomaly, not that risk is low
  - a rule that abstains says so explicitly, never rounds to a false "clean"
No template in this file fabricates a risk level the detector's own output
doesn't support.
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter

import numpy as np
import pandas as pd

from tenetdrill import detect, physics

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "volve"
RISK_SCORED_PARQUET = DATA_DIR / "risk_scored.parquet"
OUT_PATH = pathlib.Path(__file__).resolve().parent / "tenetdrill_sft.jsonl"

DEPTH_COL = physics.DEPTH_COL
GROUND_TRUTH_COL = physics.GROUND_TRUTH_COL
FUSION_RULE_NAMES = detect.FUSION_RULE_NAMES

WELL_NAME = "Norway NA 15/47-9-F-9 A"

# Anchor spacing along the well. ~4 m gives ~230 anchors across this well's
# ~933 m logged range; at 6 question templates per anchor that lands in the
# 1000-2000 pair target without hand-picking an anchor count.
ANCHOR_STEP_M = 4.0

# Lookback distance for the trend template -- roughly one EWMA span
# (detect.EWMA_SPAN_M), so the comparison reflects the same window the
# tracker itself is smoothing over.
TREND_LOOKBACK_M = 12.0

SYSTEM_PROMPT = (
    f"You are TENETDrill, an on-premise drilling advisor fine-tuned exclusively on telemetry from well "
    f"{WELL_NAME}. Answer only from this well's actual sensor readings and TENETDrill's physics-rule "
    "outputs. If a sensor is missing or a signal is inconclusive, say so plainly instead of guessing."
)

RULE_ACTIONS = detect.RULE_ACTIONS  # shared with supervisor/agent.py -- see detect.py docstring

TORQUE_QUESTIONS = [
    "Why is torque behaving the way it is at {depth}m?",
    "What's driving the torque reading at {depth}m?",
    "Torque looks off at {depth}m -- what's going on?",
]
WOB_QUESTIONS = [
    "What's the WOB/hookload trend at {depth}m? Any drag signature?",
    "Are we seeing any overpull or weight loss around {depth}m?",
    "How does hookload compare to WOB at {depth}m?",
]
SPP_QUESTIONS = [
    "What's standpipe pressure looking like at {depth}m?",
    "Any pressure anomaly at {depth}m?",
    "Is SPP behaving normally at {depth}m?",
]
RISK_QUESTIONS = [
    "Is stuck-pipe risk elevated at {depth}m?",
    "How would you rate stuck-pipe risk at {depth}m?",
    "Should I be concerned about getting stuck around {depth}m?",
]
ACTION_QUESTIONS = [
    "What should I do at {depth}m?",
    "What's your recommendation at {depth}m?",
    "Given current conditions at {depth}m, what's the play?",
]
SECONDARY_QUESTIONS = [
    "Anything notable in mud density or flow at {depth}m?",
    "How do mud density and flow/pressure consistency look at {depth}m?",
]
TREND_QUESTIONS = [
    "Has stuck-pipe risk been trending up over the last few meters approaching {depth}m?",
    "What's the risk trend heading into {depth}m?",
]


def fmt_depth(d: float) -> str:
    return f"{d:.1f}"


risk_label = detect.risk_label  # shared with supervisor/agent.py -- see detect.py docstring


def answer_torque(row: pd.Series) -> str:
    return str(row["torque_anomaly_reason"])


def answer_spp(row: pd.Series) -> str:
    return str(row["spp_spike_reason"])


def answer_wob(row: pd.Series) -> str:
    text = str(row["wob_hookload_imbalance_reason"])
    if not bool(row["wob_hookload_imbalance_abstained"]):
        text += (
            " This is currently the primary physical driver behind TENETDrill's composite risk "
            "score on this well."
        )
    return text


def answer_risk(row: pd.Series) -> str:
    depth_s = fmt_depth(row[DEPTH_COL])
    if int(row["n_fusion_active_rules"]) == 0 or pd.isna(row["fused_risk"]):
        return (
            f"I can't compute a reliable composite risk score at {depth_s}m -- all three physics rules "
            "behind the fused score (torque, WOB/hookload, standpipe pressure) are missing data at this "
            "depth. Treat this depth as unmonitored rather than assuming it's clean."
        )

    risk = float(row["ewma_risk"])
    label = risk_label(risk)
    reasons = detect.fusion_reasons(row)

    if label == "low":
        text = f"Stuck-pipe risk is low at {depth_s}m (composite trend score {risk:.2f})."
        if reasons:
            text += " Minor signal noted: " + " ".join(reasons)
        else:
            text += " No strong physical anomaly detected in the fused signals (torque, WOB/hookload, SPP)."
        return text

    text = f"Stuck-pipe risk is {label} at {depth_s}m (composite trend score {risk:.2f})."
    if reasons:
        text += " " + " ".join(reasons)
    else:
        text += (
            " The composite trend is elevated from recent history even though no single fusion rule is "
            "firing hard right at this exact row -- likely carryover from a recent event nearby."
        )
    return text


def answer_action(row: pd.Series) -> str:
    depth_s = fmt_depth(row[DEPTH_COL])
    if int(row["n_fusion_active_rules"]) == 0 or pd.isna(row["fused_risk"]):
        return (
            f"Sensor coverage is insufficient at {depth_s}m to base a recommendation on TENETDrill's "
            "physics signals. Rely on driller judgment and other rig indicators here, and flag this "
            "interval for review once telemetry resumes."
        )
    fired = detect.fusion_fired_rules(row)
    if not fired:
        return f"No corrective action needed at {depth_s}m -- continue per plan and keep monitoring."
    actions = "; ".join(RULE_ACTIONS[name] for name in fired)
    return f"At {depth_s}m: {actions}."


def answer_secondary(row: pd.Series) -> str:
    mud_text = str(row["mud_density_differential_reason"])
    flow_text = str(row["flow_spp_consistency_reason"])
    note = (
        " Note: on this well, mud-density differential and flow/SPP consistency are excluded from "
        "the composite stuck-pipe score -- validation showed mud density is not a reliable predictor "
        "here, and both signals have limited sensor coverage. Treat these as supplementary "
        "observations, not risk drivers."
    )
    return f"{mud_text} {flow_text}{note}"


def answer_trend(row: pd.Series, lookback_row: pd.Series | None) -> str:
    depth_s = fmt_depth(row[DEPTH_COL])
    if lookback_row is None or pd.isna(row["ewma_risk"]) or pd.isna(lookback_row["ewma_risk"]):
        return f"Not enough trailing history to characterize a risk trend heading into {depth_s}m."
    now, before = float(row["ewma_risk"]), float(lookback_row["ewma_risk"])
    delta = now - before
    lookback_depth = fmt_depth(lookback_row[DEPTH_COL])
    if abs(delta) < 0.03:
        direction = "roughly flat"
    elif delta > 0:
        direction = "rising"
    else:
        direction = "falling"
    return (
        f"Composite risk trend is {direction} heading into {depth_s}m: {before:.2f} at {lookback_depth}m "
        f"to {now:.2f} at {depth_s}m."
    )


def _nearest_row_idx(df: pd.DataFrame, depth: float):
    return (df[DEPTH_COL] - depth).abs().idxmin()


def make_example(question: str, answer: str, row: pd.Series, intent: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "meta": {
            "depth_m": float(row[DEPTH_COL]),
            "intent": intent,
            "fused_risk": None if pd.isna(row["fused_risk"]) else float(row["fused_risk"]),
            "ewma_risk": None if pd.isna(row["ewma_risk"]) else float(row["ewma_risk"]),
        },
    }


def generate_examples(df: pd.DataFrame, step_m: float = ANCHOR_STEP_M) -> list[dict]:
    """Deterministic generation: fixed anchor grid, fixed template rotation
    (index modulo, no RNG), so re-running produces byte-identical output.
    """
    if GROUND_TRUTH_COL in df.columns:
        raise ValueError(f"{GROUND_TRUTH_COL!r} must not reach the dataset generator -- see module docstring")

    depth_min, depth_max = float(df[DEPTH_COL].min()), float(df[DEPTH_COL].max())
    anchors = np.arange(depth_min, depth_max, step_m)

    examples: list[dict] = []
    for i, target_depth in enumerate(anchors):
        row = df.loc[_nearest_row_idx(df, float(target_depth))]
        depth_s = fmt_depth(row[DEPTH_COL])

        examples.append(make_example(TORQUE_QUESTIONS[i % len(TORQUE_QUESTIONS)].format(depth=depth_s), answer_torque(row), row, "torque_anomaly"))
        examples.append(make_example(WOB_QUESTIONS[i % len(WOB_QUESTIONS)].format(depth=depth_s), answer_wob(row), row, "wob_hookload_imbalance"))
        examples.append(make_example(SPP_QUESTIONS[i % len(SPP_QUESTIONS)].format(depth=depth_s), answer_spp(row), row, "spp_spike"))
        examples.append(make_example(RISK_QUESTIONS[i % len(RISK_QUESTIONS)].format(depth=depth_s), answer_risk(row), row, "risk_assessment"))
        examples.append(make_example(ACTION_QUESTIONS[i % len(ACTION_QUESTIONS)].format(depth=depth_s), answer_action(row), row, "recommendation"))

        if i % 2 == 0:
            examples.append(
                make_example(
                    SECONDARY_QUESTIONS[i % len(SECONDARY_QUESTIONS)].format(depth=depth_s),
                    answer_secondary(row),
                    row,
                    "secondary_signals",
                )
            )
        else:
            lookback_depth = float(row[DEPTH_COL]) - TREND_LOOKBACK_M
            lookback_row = df.loc[_nearest_row_idx(df, lookback_depth)] if lookback_depth >= depth_min else None
            examples.append(
                make_example(
                    TREND_QUESTIONS[i % len(TREND_QUESTIONS)].format(depth=depth_s),
                    answer_trend(row, lookback_row),
                    row,
                    "trend",
                )
            )

    return examples


HONESTY_MARKERS = [
    "isn't available",
    "isn't logged",
    "not both logged",
    "insufficient",
    "can't compute",
    "not enough trailing history",
    "not computable",
    "no strong physical anomaly",
    "no corrective action needed",
    "unmonitored",
]


def main() -> None:
    df = pd.read_parquet(RISK_SCORED_PARQUET)
    df = df.drop(columns=[GROUND_TRUTH_COL])  # STUCK_RT never reaches generation -- see module docstring

    examples = generate_examples(df)

    print("=" * 70)
    print("TENETDrill ILM dataset generator")
    print("=" * 70)
    print(f"Total pairs generated: {len(examples):,}")

    intent_counts = Counter(ex["meta"]["intent"] for ex in examples)
    print("\nBy intent:")
    for intent, c in sorted(intent_counts.items()):
        print(f"  {intent:22s} {c:5,}")

    n_honest = sum(
        1 for ex in examples if any(marker in ex["messages"][-1]["content"].lower() for marker in HONESTY_MARKERS)
    )
    print(f"\nAnswers containing an explicit low-confidence/no-anomaly/abstain marker: {n_honest:,} ({100*n_honest/len(examples):.1f}%)")

    risk_levels = Counter(
        risk_label(ex["meta"]["ewma_risk"]) if ex["meta"]["ewma_risk"] is not None else "unmonitored"
        for ex in examples
    )
    print("\nBy underlying ewma_risk level at the grounding depth:")
    for level, c in sorted(risk_levels.items()):
        print(f"  {level:14s} {c:5,}")

    print("\nSample pairs:")
    for ex in [examples[0], examples[len(examples) // 2], examples[-1]]:
        print(f"\n  [{ex['meta']['intent']}] depth={ex['meta']['depth_m']:.1f}m")
        print(f"  Q: {ex['messages'][1]['content']}")
        print(f"  A: {ex['messages'][2]['content']}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"\nSaved {len(examples):,} pairs -> {OUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
