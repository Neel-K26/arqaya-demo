"""
STEP 7 (backend) — FastAPI service for the TENETDrill dashboard.

Reads data/volve/{clean_depth,risk_scored}.parquet and calls
supervisor.agent.supervisor_decision() directly -- no ML model, no GPU, no
external API. Every endpoint here is deterministic Python running on the
validated Step 2/3 physics+EWMA pipeline.

/api/copilot is NOT a language model. It's a deterministic depth parser
that routes into supervisor_decision() and templates the reply -- see the
module docstring in supervisor/agent.py for the (currently-unused)
phrase_fn hook that's where a real local model would slot in later.
"""
from __future__ import annotations

import pathlib
import re

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ilm.dataset.generate import WELL_NAME
from supervisor import agent as supervisor
from tenetdrill import detect

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "volve"
CLEAN_DEPTH_PARQUET = DATA_DIR / "clean_depth.parquet"

DEPTH_COL = detect.DEPTH_COL
GROUND_TRUTH_COL = detect.GROUND_TRUTH_COL

RAW_TELEMETRY_COLUMNS = [
    "Weight on Bit kkgf",
    "Average Surface Torque kN.m",
    "Rate of Penetration m/h",
    "Average Rotary Speed rpm",
    "Mud Density In g/cm3",
    "Mud Density Out g/cm3",
    "Average Standpipe Pressure kPa",
    "Corrected Total Hookload kkgf",
    "Flow Pumps L/min",
    "MWD Shock Risk unitless",
]

app = FastAPI(title="TENETDrill API", description="Offline stuck-pipe risk detection for well " + WELL_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo deployment -- tighten to a specific origin list in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------

_MERGED_DF: pd.DataFrame | None = None


def get_risk_df() -> pd.DataFrame:
    return supervisor.load_risk_scored()


def get_merged_df() -> pd.DataFrame:
    """clean_depth.parquet (raw sensors) joined with risk_scored.parquet (risk detail) on depth.

    Both were produced from the same Step-1 resample grid (verified equal
    length and identical depth values), so this is a same-length column
    concat, not a fuzzy join.
    """
    global _MERGED_DF
    if _MERGED_DF is None:
        raw = pd.read_parquet(CLEAN_DEPTH_PARQUET)
        risk = get_risk_df().drop(columns=[GROUND_TRUTH_COL])  # raw's STUCK_RT is authoritative; avoid a duplicate
        if not np.array_equal(raw[DEPTH_COL].to_numpy(), risk[DEPTH_COL].to_numpy()):
            raise RuntimeError("clean_depth.parquet and risk_scored.parquet depth grids no longer match")
        merged = pd.concat([raw, risk.drop(columns=[DEPTH_COL])], axis=1)
        _MERGED_DF = merged
    return _MERGED_DF


def _safe(v):
    """NaN/NaT -> None so pandas values are always JSON-serializable."""
    if v is None:
        return None
    if isinstance(v, (float, np.floating)) and pd.isna(v):
        return None
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if pd.isna(v) if not isinstance(v, str) else False:
        return None
    return v


def _nearest_row(df: pd.DataFrame, depth: float) -> pd.Series:
    idx = (df[DEPTH_COL] - depth).abs().idxmin()
    return df.loc[idx]


# ---------------------------------------------------------------------------
# Derived, clearly-labeled proxy metrics for the overview cards
# ---------------------------------------------------------------------------


def _well_integrity_pct(df: pd.DataFrame) -> float | None:
    """Proxy: % of MONITORED depth with no elevated/high stuck-pipe risk signature.

    Not a dedicated wellbore-stability/integrity model -- this dataset has
    no casing/cement/pressure-containment telemetry to build one from.
    """
    monitored = df[df["n_fusion_active_rules"] > 0]
    if len(monitored) == 0:
        return None
    labels = monitored["ewma_risk"].apply(detect.risk_label)
    return float(100.0 * labels.isin(["low", "mild/moderate"]).mean())


def _npt_risk_episodes(df: pd.DataFrame) -> int:
    """Proxy: count of contiguous elevated/high-risk depth intervals.

    This is NOT actual non-productive time -- the dataset is depth-indexed
    with no time axis, so true NPT hours cannot be computed. This counts
    distinct risk episodes as a stand-in exposure metric.
    """
    labels = df["ewma_risk"].apply(lambda r: detect.risk_label(r) if pd.notna(r) else "unmonitored")
    is_risk = labels.isin(["elevated", "high"]).to_numpy().astype(int)
    if len(is_risk) == 0:
        return 0
    transitions = np.diff(np.concatenate(([0], is_risk)))
    return int((transitions == 1).sum())


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health():
    return {"status": "ok", "well": WELL_NAME}


@app.get("/api/overview")
def get_overview():
    df = get_risk_df()
    min_depth = float(df[DEPTH_COL].min())
    current_depth = float(df[DEPTH_COL].max())

    decision = supervisor.supervisor_decision(current_depth, df=df)

    return {
        "well": WELL_NAME,
        "current_depth_m": current_depth,
        "total_depth_m": current_depth,
        "depth_range_m": [min_depth, current_depth],
        "stuck_pipe_risk": {
            "score": decision["risk_score"],
            "level": decision["risk_level"],
        },
        "well_integrity_pct": _well_integrity_pct(df),
        "well_integrity_note": (
            "Proxy metric: % of monitored depth with no elevated/high stuck-pipe risk signature. "
            "Not a dedicated wellbore-integrity model."
        ),
        "npt_events": _npt_risk_episodes(df),
        "npt_note": (
            "Proxy metric: count of contiguous elevated/high-risk depth intervals. This dataset has "
            "no time axis, so actual non-productive time in hours cannot be computed."
        ),
        "generated_by": decision["generated_by"],
    }


@app.get("/api/risk-over-depth")
def get_risk_over_depth(max_points: int = Query(800, ge=10, le=20000)):
    df = get_risk_df()
    n = len(df)
    stride = max(1, n // max_points)
    sampled = df.iloc[::stride]

    return {
        "depth_m": sampled[DEPTH_COL].round(3).tolist(),
        "fused_risk": [_safe(v) for v in sampled["fused_risk"]],
        "ewma_risk": [_safe(v) for v in sampled["ewma_risk"]],
        "stuck_rt": [_safe(v) for v in sampled[GROUND_TRUTH_COL]],
        "n_points": int(len(sampled)),
        "n_total_rows": int(n),
        "note": "stuck_rt is the historical ground-truth label shown for retrospective validation only -- it is never used by the live risk score or the supervisor agent.",
    }


@app.get("/api/telemetry")
def get_telemetry(depth: float = Query(..., description="Measured depth in meters")):
    df = get_merged_df()
    if len(df) == 0:
        raise HTTPException(status_code=500, detail="telemetry data not loaded")

    row = _nearest_row(df, depth)
    return {
        "requested_depth_m": depth,
        "depth_m": float(row[DEPTH_COL]),
        "telemetry": {col: _safe(row[col]) for col in RAW_TELEMETRY_COLUMNS},
        "risk": {
            "fused_risk": _safe(row["fused_risk"]),
            "ewma_risk": _safe(row["ewma_risk"]),
            "rules_fired": row["rules_fired"] if row["rules_fired"] else None,
        },
        "stuck_rt_ground_truth": _safe(row[GROUND_TRUTH_COL]),
    }


@app.get("/api/alert")
def get_alert(depth: float | None = Query(None, description="Defaults to the deepest logged depth")):
    df = get_risk_df()
    if depth is None:
        depth = float(df[DEPTH_COL].max())
    return supervisor.supervisor_decision(depth, df=df)


class CopilotRequest(BaseModel):
    message: str


_DEPTH_WITH_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*m\b", re.IGNORECASE)
_ANY_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")
_CURRENT_DEPTH_RE = re.compile(r"\b(current|now|latest|today)\b", re.IGNORECASE)


def _parse_depth_from_message(message: str, df: pd.DataFrame) -> float | None:
    match = _DEPTH_WITH_UNIT_RE.search(message) or _ANY_NUMBER_RE.search(message)
    if match:
        return float(match.group(1))
    if _CURRENT_DEPTH_RE.search(message):
        return float(df[DEPTH_COL].max())
    return None


@app.post("/api/copilot")
def post_copilot(req: CopilotRequest):
    df = get_risk_df()
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message must not be empty")

    depth = _parse_depth_from_message(message, df)
    if depth is None:
        return {
            "reply": (
                "I can answer questions about a specific depth -- try something like "
                '"is stuck-pipe risk elevated at 800m?" or "what should I do at 512m?". '
                'You can also ask about "current" conditions for the latest logged depth.'
            ),
            "depth_m": None,
            "decision": None,
        }

    min_depth, max_depth = float(df[DEPTH_COL].min()), float(df[DEPTH_COL].max())
    depth = max(min_depth, min(depth, max_depth))

    decision = supervisor.supervisor_decision(depth, df=df)
    reply = (
        f"At {decision['depth_m']:.1f}m: {decision['explanation']} "
        f"Recommended: {'; '.join(decision['recommended_actions'])}"
    )
    return {"reply": reply, "depth_m": decision["depth_m"], "decision": decision}
