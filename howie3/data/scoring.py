"""Fantasy point computation. One place, all three formats, vectorized."""

from typing import Dict

import pandas as pd

# Shared base rules; formats differ only in reception value
BASE = dict(
    pass_yds=0.04, pass_td=4.0, pass_int=-2.0,
    rush_yds=0.1, rush_td=6.0,
    rec_yds=0.1, rec_td=6.0,
    fumble_lost=-2.0, two_pt=2.0, st_td=6.0,
)
RECEPTION_VALUE = {"std": 0.0, "half": 0.5, "ppr": 1.0}


def add_points_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add pts_std/pts_half/pts_ppr to a frame with our canonical stat columns.

    Missing stat columns are treated as zero (e.g. projections without st_tds).
    """
    def col(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=df.index)

    base_pts = (
        BASE["pass_yds"] * col("pass_yards")
        + BASE["pass_td"] * col("pass_tds")
        + BASE["pass_int"] * col("interceptions")
        + BASE["rush_yds"] * col("rush_yards")
        + BASE["rush_td"] * col("rush_tds")
        + BASE["rec_yds"] * col("rec_yards")
        + BASE["rec_td"] * col("rec_tds")
        + BASE["fumble_lost"] * col("fumbles_lost")
        + BASE["two_pt"] * col("two_pt")
        + BASE["st_td"] * col("st_tds")
    )
    receptions = col("receptions")
    for fmt, rec_val in RECEPTION_VALUE.items():
        df[f"pts_{fmt}"] = (base_pts + rec_val * receptions).round(2)
    return df


# --- Kicker scoring (reception-independent: same in all formats) ---
KICKER_RULES = dict(fg_0_39=3.0, fg_40_49=4.0, fg_50_plus=5.0, pat=1.0)


def kicker_points(df: pd.DataFrame) -> pd.Series:
    def col(name):
        return pd.to_numeric(df.get(name), errors="coerce").fillna(0.0) if name in df.columns else 0.0

    return (
        KICKER_RULES["fg_0_39"] * (col("fg_made_0_19") + col("fg_made_20_29") + col("fg_made_30_39"))
        + KICKER_RULES["fg_40_49"] * col("fg_made_40_49")
        + KICKER_RULES["fg_50_plus"] * col("fg_made_50_plus")
        + KICKER_RULES["pat"] * col("pat_made")
    ).round(2)


# --- Team defense scoring (reception-independent) ---
DST_RULES = dict(sack=1.0, interception=2.0, fumble_rec=2.0, safety=2.0, td=6.0)
# Expected points-allowed points: games projected in each bucket x bucket value
DST_PA_BUCKET_VALUES = dict(
    dst_pa_0=10.0, dst_pa_1_6=7.0, dst_pa_7_13=4.0, dst_pa_14_20=1.0,
    dst_pa_21_27=0.0, dst_pa_28_34=-1.0, dst_pa_35_plus=-4.0,
)


def dst_points(df: pd.DataFrame) -> pd.Series:
    def col(name):
        return pd.to_numeric(df.get(name), errors="coerce").fillna(0.0) if name in df.columns else 0.0

    pts = (
        DST_RULES["sack"] * col("dst_sacks")
        + DST_RULES["interception"] * col("dst_ints")
        + DST_RULES["fumble_rec"] * col("dst_fumbles_rec")
        + DST_RULES["safety"] * col("dst_safeties")
        + DST_RULES["td"] * (col("dst_tds") + col("dst_return_tds"))
    )
    for bucket, value in DST_PA_BUCKET_VALUES.items():
        pts = pts + value * col(bucket)
    return pts.round(2)
