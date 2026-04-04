import os
import math
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


FRED_API_KEY_ENV = "FRED_API_KEY"


def fred_series_observations(series_id: str, api_key: str, units: str = "lin") -> pd.DataFrame:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "api_key": api_key,
        "file_type": "json",
        "series_id": series_id,
        "units": units,
        "sort_order": "asc",
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    obs = data.get("observations", [])
    if not obs:
        raise RuntimeError(f"No observations returned for series {series_id}")

    df = pd.DataFrame(obs)[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().set_index("date").sort_index()
    return df


def compute_10y_zscore(df: pd.DataFrame):
    """
    Returns (latest_value, mean_10y, std_10y, z_score, latest_date)
    Uses explicit cutoff (no deprecated .last("10Y")).
    """
    latest_date = df.index[-1]
    latest_value = float(df["value"].iloc[-1])

    cutoff_date = latest_date - pd.DateOffset(years=10)
    ten_years = df.loc[df.index >= cutoff_date]

    mean_10y = float(ten_years["value"].mean())
    std_10y = float(ten_years["value"].std())

    if std_10y == 0 or np.isnan(std_10y):
        raise RuntimeError("10Y std dev invalid (0 or NaN), cannot compute z-score.")

    z = (latest_value - mean_10y) / std_10y
    return latest_value, mean_10y, std_10y, z, latest_date


def draw_dial(
    value: float,
    *,
    min_val: float,
    max_val: float,
    title: str,
    state: str,
    outfile: str,
    value_suffix: str = "",
    tick_step: float = 1.0,
) -> None:
    ARC_COLOR = "#000000"
    NEEDLE_COLOR = "#968C83"
    CENTER_COLOR = "#000000"

    v = max(min_val, min(value, max_val))
    frac = (v - min_val) / (max_val - min_val) if max_val > min_val else 0.0
    angle = math.pi * (1 - frac)

    fig, ax = plt.subplots(figsize=(7, 4))

    # Main arc
    theta = np.linspace(0, math.pi, 500)
    ax.plot(np.cos(theta), np.sin(theta), linewidth=10, color=ARC_COLOR)

    # Tick marks placed at actual numeric values
    tick_values = np.arange(min_val, max_val + 1e-9, tick_step)

    for tval in tick_values:
        tfrac = (tval - min_val) / (max_val - min_val) if max_val > min_val else 0.0
        tang = math.pi * (1 - tfrac)

        x0, y0 = 0.92 * math.cos(tang), 0.92 * math.sin(tang)
        x1, y1 = 1.02 * math.cos(tang), 1.02 * math.sin(tang)
        ax.plot([x0, x1], [y0, y1], linewidth=2, color="black")

        # Cleaner formatting for integer vs non-integer ticks
        if abs(tval - round(tval)) < 1e-9:
            tick_label = f"{int(round(tval))}{value_suffix}"
        else:
            tick_label = f"{tval:.1f}{value_suffix}"

        ax.text(
            1.17 * math.cos(tang),
            1.17 * math.sin(tang),
            tick_label,
            ha="center",
            va="center",
            fontsize=10,
            color="black",
        )

    # Needle
    ax.plot(
        [0, 0.85 * math.cos(angle)],
        [0, 0.85 * math.sin(angle)],
        linewidth=6,
        color=NEEDLE_COLOR,
    )
    ax.scatter([0], [0], s=250, color=CENTER_COLOR)

    # Labels
    ax.text(0, 1.25, title, ha="center", fontsize=16, fontweight="bold", color="black")
    ax.text(0, -0.25, f"{v:.2f}{value_suffix}", ha="center", fontsize=24, fontweight="bold", color="black")
    ax.text(0, -0.45, f"Regime: {state}", ha="center", fontsize=14, fontweight="bold", color="black")

    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.5, 1.35)
    ax.axis("off")

    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Regime classifiers (z-score based)
# -----------------------------
def inflation_regime_from_z(z: float) -> str:
    if z < -0.5:
        return "Cooling"
    elif z < 0.5:
        return "Sticky"
    else:
        return "Rising"


def credit_regime_from_z(z: float) -> str:
    # Higher spreads = worse
    if z < -0.5:
        return "Calm"
    elif z < 0.5:
        return "Cautious"
    else:
        return "Stressed"


def rates_regime_from_z(z: float) -> str:
    # Higher rates = more restrictive (generally)
    if z < -0.5:
        return "Supportive"
    elif z < 0.5:
        return "Neutral"
    else:
        return "Restrictive"


def main():
    api_key = os.getenv(FRED_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{FRED_API_KEY_ENV} not found. Set it and restart your terminal.")

    outputs = []

    # =====================================================
    # 1) Inflation Dial (CPI YoY)
    # =====================================================
    cpi = fred_series_observations("CPIAUCSL", api_key=api_key, units="pc1")
    cpi_latest, cpi_mean, cpi_std, cpi_z, cpi_date = compute_10y_zscore(cpi)
    cpi_state = inflation_regime_from_z(cpi_z)

    cpi_out = os.path.abspath("inflation_dial.png")
    draw_dial(
        cpi_latest,
        min_val=0.0,
        max_val=6.0,
        title="Inflation Backdrop",
        state=cpi_state,
        outfile=cpi_out,
        value_suffix="%",
        tick_step=1.0,
    )
    print(f"Saved: {cpi_out}")
    outputs.append(cpi_out)

    # =====================================================
    # 2) Credit Spreads Dial (HY OAS)
    # =====================================================
    # BAMLH0A0HYM2 is in percent (e.g., 3.50 means 3.50% = 350 bps)
    hy = fred_series_observations("BAMLH0A0HYM2", api_key=api_key, units="lin")
    hy_latest, hy_mean, hy_std, hy_z, hy_date = compute_10y_zscore(hy)
    hy_state = credit_regime_from_z(hy_z)

    hy_out = os.path.abspath("credit_spreads_dial.png")
    draw_dial(
        hy_latest,
        min_val=2.0,
        max_val=10.0,
        title="Credit Spreads (High Yield OAS)",
        state=hy_state,
        outfile=hy_out,
        value_suffix="%",
        tick_step=1.0,
    )
    print(f"Saved: {hy_out}")
    outputs.append(hy_out)

    # =====================================================
    # 3) Rates Dial (10Y Treasury Yield)
    # =====================================================
    dgs10 = fred_series_observations("DGS10", api_key=api_key, units="lin")
    r_latest, r_mean, r_std, r_z, r_date = compute_10y_zscore(dgs10)
    r_state = rates_regime_from_z(r_z)

    rates_out = os.path.abspath("rates_dial.png")
    draw_dial(
        r_latest,
        min_val=0.0,
        max_val=6.0,
        title="Rates Backdrop (10Y Treasury)",
        state=r_state,
        outfile=rates_out,
        value_suffix="%",
        tick_step=1.0,
    )
    print(f"Saved: {rates_out}")
    outputs.append(rates_out)

    # Auto-open all (Windows)
    try:
        for p in outputs:
            os.startfile(p)
    except Exception as e:
        print(f"Could not auto-open images: {e}")


if __name__ == "__main__":
    main()