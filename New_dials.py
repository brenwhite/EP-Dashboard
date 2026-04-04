import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===========================
# CONFIG
# ===========================
OUT_DIR = "dials"
TICKER_DIR = os.path.join(OUT_DIR, "tickers")
CLASS_DIR = os.path.join(OUT_DIR, "classifications")

# Column name for grouping
CLASS_COL = "Classification"

# Render options
USE_ALL_TICKERS_IN_CSV = True
TICKERS = ["SPY", "SCHF", "EEM", "GLD", "GSG"]  # only used if USE_ALL_TICKERS_IN_CSV=False

# Master weights
W_TREND = 0.40
W_MOM = 0.30
W_RISK = 0.20
W_PART = 0.10

# Participation weights across horizons
P_W = 0.40
P_M = 0.35
P_Y = 0.25

# Dial style
ARC_COLOR = "#000000"
NEEDLE_COLOR = "#968C83"
CENTER_COLOR = "#000000"
SHOW_SUBSCORES = False


# ===========================
# FILE HELPERS
# ===========================
def find_trend_csv() -> str:
    candidates = glob.glob("Trend.csv") + glob.glob("trend.csv") + glob.glob("TREND.csv")
    if candidates:
        return candidates[0]
    candidates = glob.glob("*trend*.csv")
    if candidates:
        return candidates[0]
    raise FileNotFoundError("Could not find Trend.csv (or any *trend*.csv) in the current folder.")


def pick_col(df: pd.DataFrame, preferred: str, fallback: str) -> str:
    if preferred in df.columns:
        return preferred
    if fallback in df.columns:
        return fallback
    raise ValueError(f"Missing required column: '{preferred}' (or fallback '{fallback}')")


# ===========================
# SCORING UTILITIES
# ===========================
def linear_score(value, low: float, high: float):
    """Vectorized 0..100 map for scalar or Series."""
    if high == low:
        if isinstance(value, pd.Series):
            return pd.Series(np.full(len(value), 50.0), index=value.index)
        return 50.0

    arr = pd.to_numeric(value, errors="coerce")
    frac = (arr - low) / (high - low)
    score = np.clip(frac * 100.0, 0.0, 100.0)

    if isinstance(value, pd.Series):
        return pd.Series(score, index=value.index)
    return float(score)


def dist_score(price: float, ema: float, low: float = -0.10, high: float = 0.10) -> float:
    """Scalar distance-to-EMA score."""
    if pd.isna(price) or pd.isna(ema) or ema == 0:
        return np.nan
    dist = (price - ema) / ema
    return float(np.clip(((dist - low) / (high - low)) * 100.0, 0.0, 100.0))


def score_rsi(val: float) -> float:
    """RSI score 0..100 favoring 50-65, penalizing <40 and >70."""
    if pd.isna(val):
        return np.nan

    if val < 30:
        score = 15
    elif val < 40:
        score = 15 + (val - 30) * 3.0
    elif val < 50:
        score = 45 + (val - 40) * 3.5
    elif val < 65:
        score = 80 + (val - 50) * (20 / 15)
    elif val < 70:
        score = 100 - (val - 65) * 3.0
    else:
        score = 85 - (val - 70) * 4.0

    return float(np.clip(score, 0, 100))


def safe_pct_series(s: pd.Series) -> pd.Series:
    """
    Normalize percent-ish columns:
      - if median abs > 1 assume % units like 5.2 -> convert to 0.052
      - else assume already decimal like 0.052
    """
    s2 = pd.to_numeric(s, errors="coerce")
    med = s2.abs().median(skipna=True)
    if pd.notna(med) and med > 1:
        return s2 / 100.0
    return s2


def flow_percentile_scores(df: pd.DataFrame, flow_col: str, tickers: list[str]) -> pd.Series:
    """Cross-sectional percentile rank of flows among tickers."""
    sub = df[df["Ticker"].isin(tickers)].copy()
    flows = pd.to_numeric(sub[flow_col], errors="coerce")

    if flows.notna().sum() == 0:
        return pd.Series(index=df.index, data=np.nan, dtype=float)

    pct = flows.rank(pct=True) * 100.0
    out = pd.Series(index=df.index, data=np.nan, dtype=float)
    out.loc[sub.index] = pct.values
    return out


# ===========================
# LABELS
# ===========================
def regime_from_score(score: float) -> str:
    if pd.isna(score):
        return "No Data"
    if score >= 80:
        return "Strong Bull"
    if score >= 60:
        return "Bull"
    if score >= 40:
        return "Neutral"
    if score >= 20:
        return "Bear"
    return "Strong Bear"


# ===========================
# DIAL RENDERER
# ===========================
def draw_dial(
    value: float,
    *,
    min_val: float,
    max_val: float,
    title: str,
    state: str,
    outfile: str,
    tick_format: str = "{:.0f}",
    tick_suffix: str = "",
    value_suffix: str = "",
    subtext: str | None = None,
) -> None:
    v = max(min_val, min(value, max_val)) if pd.notna(value) else (min_val + max_val) / 2
    frac = (v - min_val) / (max_val - min_val) if max_val > min_val else 0.0
    angle = math.pi * (1 - frac)

    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)

    theta = np.linspace(0, math.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), linewidth=10, color=ARC_COLOR, solid_capstyle="round")

    ticks = 6
    for i in range(ticks + 1):
        tfrac = i / ticks
        tang = math.pi * (1 - tfrac)

        x0, y0 = 0.95 * math.cos(tang), 0.95 * math.sin(tang)
        x1, y1 = 1.01 * math.cos(tang), 1.01 * math.sin(tang)
        ax.plot([x0, x1], [y0, y1], linewidth=1.8, color="black")

        tval = min_val + tfrac * (max_val - min_val)
        ax.text(
            1.14 * math.cos(tang),
            1.14 * math.sin(tang),
            f"{tick_format.format(tval)}{tick_suffix}",
            ha="center",
            va="center",
            fontsize=10,
            color="black",
        )

    ax.plot([0, 0.82 * math.cos(angle)], [0, 0.82 * math.sin(angle)], linewidth=5.5, color=NEEDLE_COLOR)
    ax.scatter([0], [0], s=220, color=CENTER_COLOR)

    ax.text(0, 1.32, title, ha="center", fontsize=18, fontweight="bold", color="black")
    val_str = "NA" if pd.isna(value) else f"{v:.1f}{value_suffix}"
    ax.text(0, -0.22, val_str, ha="center", fontsize=30, fontweight="bold", color="black")

    if subtext:
        ax.text(0, -0.40, subtext, ha="center", fontsize=10, color="#333333")

    ax.text(0, -0.58, f"Regime: {state}", ha="center", fontsize=14, fontweight="bold", color="black")

    ax.set_aspect("equal")
    ax.axis("off")

    plt.savefig(outfile, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ===========================
# SCORING PIPELINE (TICKERS)
# ===========================
def compute_ticker_scores(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    # Required columns (core to the logic)
    required = [
        "Ticker", "Name", "Last Price",
        "Below 52W High %",
        "RSI", "RSI 10D",
        "Volatility (1M)", "Volatility (3M)", "Volatility (6M)", "Volatility (1Y)",
        "Fund Flows/Periodic (W)", "Fund Flows/Periodic (M)", "Fund Flows/Periodic (Y)",
        "EMA (5D)", "EMA (50D)", "EMA (200D)",
        "Price Chg. % (1W)", "Price Chg. % (1M)", "Price Chg. % (3M)",
        "Price Chg. % (6M)", "Price Chg. % (1Y)",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Trend.csv missing required column(s): {missing}")

    ema_short = pick_col(df, "EMA (21D)", "EMA (20D)")

    # Optional long EMAs
    optional_emas = [c for c in ["EMA (100D)", "EMA (150D)", "EMA (250D)", "EMA (300D)"] if c in df.columns]

    # Coerce numerics
    for c in df.columns:
        if c not in ("Ticker", "Name", CLASS_COL):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Normalize percent columns
    pct_cols = [
        "Below 52W High %",
        "Price Chg. % (1W)", "Price Chg. % (1M)", "Price Chg. % (3M)",
        "Price Chg. % (6M)", "Price Chg. % (1Y)",
        "Price Chg. % (3Y)", "Price Chg. % (5Y)", "Price Chg. % (10Y)",
    ]
    for c in pct_cols:
        if c in df.columns:
            df[c] = safe_pct_series(df[c])

    # -----------------------
    # TREND
    # -----------------------
    ema_list = ["EMA (5D)", ema_short, "EMA (50D)"] + optional_emas + ["EMA (200D)"]
    # Ensure 200D is last (if 250/300 exist, keep them after 200 for ordering and distance anchoring)
    if "EMA (200D)" in ema_list:
        ema_list = [e for e in ema_list if e != "EMA (200D)"] + ["EMA (200D)"]
    # If 250/300 exist, append after 200 for extra anchors
    for extra in ["EMA (250D)", "EMA (300D)"]:
        if extra in df.columns and extra not in ema_list:
            ema_list.append(extra)

    above_votes = [(df["Last Price"] > df[e]).astype(float) for e in ema_list]
    price_above_score = (sum(above_votes) / len(above_votes)) * 100.0

    order_votes = [(df[a] > df[b]).astype(float) for a, b in zip(ema_list[:-1], ema_list[1:])]
    ema_order_score = (sum(order_votes) / len(order_votes)) * 100.0 if order_votes else 50.0

    stack_score = 0.60 * price_above_score + 0.40 * ema_order_score

    # Distance to anchors (50/200 + optional 300)
    d50 = df.apply(lambda x: dist_score(x["Last Price"], x["EMA (50D)"]), axis=1)
    d200 = df.apply(lambda x: dist_score(x["Last Price"], x["EMA (200D)"]), axis=1)
    dist_components = [d50, d200]

    if "EMA (300D)" in df.columns:
        d300 = df.apply(lambda x: dist_score(x["Last Price"], x["EMA (300D)"]), axis=1)
        dist_components.append(d300)

    dist_long_score = pd.concat(dist_components, axis=1).mean(axis=1, skipna=True)
    df["Trend_Score"] = 0.50 * stack_score + 0.50 * dist_long_score

    # -----------------------
    # MOMENTUM (RSI + cross + multi-horizon returns)
    # -----------------------
    rsi_score = df["RSI"].apply(score_rsi)
    rsi10_score = df["RSI 10D"].apply(score_rsi)
    cross_score = np.where(df["EMA (5D)"] > df[ema_short], 100.0, 0.0)

    p1w = linear_score(df["Price Chg. % (1W)"], low=-0.05, high=0.05)
    p1m = linear_score(df["Price Chg. % (1M)"], low=-0.10, high=0.10)
    p3m = linear_score(df["Price Chg. % (3M)"], low=-0.15, high=0.15)
    p6m = linear_score(df["Price Chg. % (6M)"], low=-0.25, high=0.25)
    p1y = linear_score(df["Price Chg. % (1Y)"], low=-0.40, high=0.40)

    price_mom_score = 0.20 * p1w + 0.25 * p1m + 0.25 * p3m + 0.20 * p6m + 0.10 * p1y

    df["Momentum_Score"] = 0.35 * rsi_score + 0.25 * rsi10_score + 0.15 * cross_score + 0.25 * price_mom_score

    # -----------------------
    # RISK
    # -----------------------
    v1 = df["Volatility (1M)"]
    v3 = df["Volatility (3M)"]
    v6 = df["Volatility (6M)"]
    vY = df["Volatility (1Y)"]

    worse_count = (v1 > vY).astype(float) + (v3 > vY).astype(float) + (v6 > vY).astype(float)
    vol_regime_score = worse_count.map({0.0: 90.0, 1.0: 70.0, 2.0: 45.0, 3.0: 20.0})

    drawdown_score = linear_score(df["Below 52W High %"], low=-0.20, high=0.0)
    df["Risk_Score"] = 0.70 * vol_regime_score + 0.30 * drawdown_score

    # -----------------------
    # PARTICIPATION (relative across tickers, blended W/M/Y)
    # -----------------------
    part_w = flow_percentile_scores(df, "Fund Flows/Periodic (W)", tickers)
    part_m = flow_percentile_scores(df, "Fund Flows/Periodic (M)", tickers)
    part_y = flow_percentile_scores(df, "Fund Flows/Periodic (Y)", tickers)
    df["Participation_Score"] = P_W * part_w + P_M * part_m + P_Y * part_y

    # -----------------------
    # MASTER
    # -----------------------
    df["Master_Score"] = (
        W_TREND * df["Trend_Score"] +
        W_MOM * df["Momentum_Score"] +
        W_RISK * df["Risk_Score"] +
        W_PART * df["Participation_Score"]
    )

    return df


# ===========================
# CLASSIFICATION AGGREGATION
# ===========================
def compute_classification_scores(df_scored: pd.DataFrame) -> pd.DataFrame:
    if CLASS_COL not in df_scored.columns:
        raise ValueError(f"Missing '{CLASS_COL}' column. Add it to Trend.csv or change CLASS_COL.")

    grp = df_scored.groupby(CLASS_COL, dropna=False)

    # Average the components across tickers in each classification
    out = grp[["Trend_Score", "Momentum_Score", "Risk_Score", "Participation_Score", "Master_Score"]].mean()

    # Count tickers per classification
    out["Count"] = grp.size()

    # Recompute Master from averaged components (keeps weights consistent)
    out["Master_Score"] = (
        W_TREND * out["Trend_Score"] +
        W_MOM * out["Momentum_Score"] +
        W_RISK * out["Risk_Score"] +
        W_PART * out["Participation_Score"]
    )

    out = out.reset_index().rename(columns={CLASS_COL: "Classification"})
    return out


# ===========================
# RENDERING
# ===========================
def render_ticker_dials(df_scored: pd.DataFrame, tickers: list[str]):
    for t in tickers:
        sub = df_scored[df_scored["Ticker"] == t]
        if sub.empty:
            print(f"[WARN] {t} not found. Skipping.")
            continue

        row = sub.iloc[0]
        title = str(row["Name"]) if pd.notna(row.get("Name")) else t
        master = float(row["Master_Score"]) if pd.notna(row["Master_Score"]) else np.nan
        state = regime_from_score(master)

        subtext = None
        if SHOW_SUBSCORES:
            def f0(x): return "NA" if pd.isna(x) else f"{float(x):.0f}"
            subtext = (
                f"Trend {f0(row['Trend_Score'])} | "
                f"Mom {f0(row['Momentum_Score'])} | "
                f"Risk {f0(row['Risk_Score'])} | "
                f"Part {f0(row['Participation_Score'])}"
            )

        out_path = os.path.abspath(os.path.join(TICKER_DIR, f"{t}_dial.png"))
        draw_dial(
            master,
            min_val=0.0,
            max_val=100.0,
            title=title,
            state=state,
            outfile=out_path,
            tick_format="{:.0f}",
            subtext=subtext
        )
        print(f"Saved: {out_path}")


def render_classification_dials(df_class: pd.DataFrame):
    for _, row in df_class.iterrows():
        label = str(row["Classification"])
        master = float(row["Master_Score"]) if pd.notna(row["Master_Score"]) else np.nan
        state = regime_from_score(master)

        subtext = None
        if SHOW_SUBSCORES:
            def f0(x): return "NA" if pd.isna(x) else f"{float(x):.0f}"
            subtext = (
                f"Trend {f0(row['Trend_Score'])} | "
                f"Mom {f0(row['Momentum_Score'])} | "
                f"Risk {f0(row['Risk_Score'])} | "
                f"Part {f0(row['Participation_Score'])} | "
                f"N={int(row['Count'])}"
            )

        out_path = os.path.abspath(os.path.join(CLASS_DIR, f"{label}_dial.png".replace("/", "-")))
        draw_dial(
            master,
            min_val=0.0,
            max_val=100.0,
            title=label,
            state=state,
            outfile=out_path,
            tick_format="{:.0f}",
            subtext=subtext
        )
        print(f"Saved: {out_path}")


# ===========================
# MAIN
# ===========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TICKER_DIR, exist_ok=True)
    os.makedirs(CLASS_DIR, exist_ok=True)

    csv_path = find_trend_csv()
    df = pd.read_csv(csv_path)

    tickers = sorted(df["Ticker"].dropna().unique().tolist()) if USE_ALL_TICKERS_IN_CSV else TICKERS

    # Score tickers
    df_scored = compute_ticker_scores(df, tickers)

    # Save ticker-level output
    df_scored.to_csv("Scored_Technical_Dials.csv", index=False)
    print("Saved: Scored_Technical_Dials.csv")

    # Render ticker dials
    render_ticker_dials(df_scored, tickers)

    # Build + save classification-level table
    df_class = compute_classification_scores(df_scored)
    df_class.to_csv("Scored_Classification_Dials.csv", index=False)
    print("Saved: Scored_Classification_Dials.csv")

    # Render classification dials
    render_classification_dials(df_class)


if __name__ == "__main__":
    main()
