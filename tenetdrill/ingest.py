"""
STEP 1 — Ingestion for TENETDrill.

Loads the Volve depth-indexed drilling telemetry CSV, selects the key
columns, cleans NaN gaps and physically-impossible outliers, and writes a
clean depth-indexed parquet file for downstream physics/detection/ILM steps.

Usage:
    python -m tenetdrill.ingest
    python tenetdrill/ingest.py
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "volve"
RAW_CSV = DATA_DIR / "Norway-NA-15_47_9-F-9 A depth.csv"
CLEAN_PARQUET = DATA_DIR / "clean_depth.parquet"

DEPTH_COL = "Measured Depth m"

# The key telemetry columns this system is built on. Exact names as they
# appear in the source CSV.
KEY_COLUMNS = [
    DEPTH_COL,
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
    "STUCK_RT unitless",
]

# Physically-reasonable bounds used to null out sensor/logging garbage.
# Values outside these ranges are set to NaN rather than dropped, so the
# depth index stays intact for resampling.
PHYSICAL_BOUNDS = {
    # Negative WOB means the bit is being pulled, not drilled -- common
    # off-bottom/tripping artifact in this file, not real weight-on-bit.
    "Weight on Bit kkgf": (0, 60),
    # Surface torque cannot be meaningfully negative under rotation.
    "Average Surface Torque kN.m": (0, 100),
    # Sustained ROP above ~300 m/h is not physically achievable rotary
    # drilling; the file contains spikes up to 21200 m/h from
    # instantaneous-rate calculation artifacts during connections.
    "Rate of Penetration m/h": (0, 300),
    "Average Rotary Speed rpm": (0, 400),
    "Mud Density In g/cm3": (0.5, 3.0),
    "Mud Density Out g/cm3": (0.5, 3.0),
    "Average Standpipe Pressure kPa": (0, 60000),
    "Corrected Total Hookload kkgf": (0, 500),
    "Flow Pumps L/min": (0, 10000),
    "MWD Shock Risk unitless": (0, 10),
}

STUCK_COL = "STUCK_RT unitless"


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, usecols=KEY_COLUMNS)
    df = df.sort_values(DEPTH_COL).reset_index(drop=True)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    outlier_counts = {}

    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        mask = (df[col] < lo) | (df[col] > hi)
        outlier_counts[col] = int(mask.sum())
        df.loc[mask, col] = np.nan

    # Drop exact-duplicate depth rows (none expected, but guard anyway).
    before = len(df)
    df = df.drop_duplicates(subset=[DEPTH_COL]).reset_index(drop=True)
    dupes_dropped = before - len(df)

    return df, outlier_counts, dupes_dropped


def resample(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate telemetry onto a clean, evenly-spaced depth index.

    STUCK_RT is a discrete ground-truth flag, so it is forward/backward
    filled (nearest known label) rather than interpolated, and left NaN
    outside the labeled interval instead of being fabricated.
    """
    depth = df[DEPTH_COL].to_numpy()
    step = np.median(np.diff(depth))
    step = round(step, 3)

    grid = np.round(
        np.arange(depth.min(), depth.max() + step / 2, step), 3
    )

    indexed = df.set_index(DEPTH_COL)
    indexed = indexed[~indexed.index.duplicated(keep="first")]

    out = pd.DataFrame(index=grid)
    out.index.name = DEPTH_COL

    value_cols = [c for c in KEY_COLUMNS if c not in (DEPTH_COL, STUCK_COL)]
    for col in value_cols:
        series = indexed[col].reindex(indexed.index.union(grid)).sort_index()
        series = series.interpolate(method="index", limit_area="inside")
        out[col] = series.reindex(grid).to_numpy()

    # Nearest-known-label fill for the discrete stuck-pipe flag, restricted
    # to the depth interval where it was actually logged.
    stuck = indexed[STUCK_COL].dropna()
    if len(stuck):
        stuck_reindexed = stuck.reindex(stuck.index.union(grid)).sort_index()
        stuck_filled = stuck_reindexed.ffill().bfill()
        out[STUCK_COL] = stuck_filled.reindex(grid).to_numpy()
        out.loc[out.index < stuck.index.min(), STUCK_COL] = np.nan
        out.loc[out.index > stuck.index.max(), STUCK_COL] = np.nan
    else:
        out[STUCK_COL] = np.nan

    out = out.reset_index()
    return out


def summarize(raw: pd.DataFrame, clean_df: pd.DataFrame, outlier_counts: dict, dupes_dropped: int) -> None:
    print("=" * 70)
    print("TENETDrill Ingestion Summary")
    print("=" * 70)
    print(f"Source: {RAW_CSV.name}")
    print(f"Raw rows: {len(raw):,}")
    print(f"Resampled rows: {len(clean_df):,}")
    print(
        f"Depth range: {clean_df[DEPTH_COL].min():.2f} m - "
        f"{clean_df[DEPTH_COL].max():.2f} m"
    )
    step = clean_df[DEPTH_COL].diff().median()
    print(f"Depth step (resampled): {step:.3f} m")
    print(f"Duplicate-depth rows dropped: {dupes_dropped}")

    print("\nOutliers nulled per physical bounds check:")
    for col, n in outlier_counts.items():
        if n:
            lo, hi = PHYSICAL_BOUNDS[col]
            print(f"  {col:38s} {n:6d} rows outside [{lo}, {hi}]")

    print("\nSTUCK_RT events by level (resampled grid, within labeled interval):")
    vc = clean_df[STUCK_COL].value_counts(dropna=True).sort_index()
    total_labeled = int(vc.sum())
    for level, count in vc.items():
        pct = 100 * count / total_labeled if total_labeled else 0
        print(f"  level {int(level)}: {count:6d} rows ({pct:5.1f}%)")
    print(f"  (unlabeled / outside logged interval: {clean_df[STUCK_COL].isna().sum()} rows)")

    print("\nParameter ranges (post-clean, resampled):")
    param_cols = [c for c in KEY_COLUMNS if c not in (DEPTH_COL, STUCK_COL)]
    stats = clean_df[param_cols].describe().T[["min", "mean", "max"]]
    with pd.option_context("display.float_format", "{:.2f}".format):
        print(stats)

    print("\nNaN coverage (post-clean, resampled):")
    na_pct = (clean_df[param_cols + [STUCK_COL]].isna().mean() * 100).round(1)
    for col, pct in na_pct.items():
        print(f"  {col:38s} {pct:5.1f}% NaN")
    print("=" * 70)


def main() -> None:
    raw = load_raw()
    cleaned, outlier_counts, dupes_dropped = clean(raw)
    resampled = resample(cleaned)

    summarize(raw, resampled, outlier_counts, dupes_dropped)

    CLEAN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    resampled.to_parquet(CLEAN_PARQUET, index=False)
    print(f"\nSaved clean parquet -> {CLEAN_PARQUET}")


if __name__ == "__main__":
    main()
