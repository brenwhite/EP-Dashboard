import os
import math
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


FRED_API_KEY_ENV = "FRED_API_KEY"

ARC_COLOR = "#000000"
NEEDLE_COLOR = "#968C83"
CENTER_COLOR = "#000000"

OUTPUT_DIR = "."


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

    if df.empty:
        raise RuntimeError(f"Series {series_id} became empty after cleaning (all missing?).")

    return df


def resample_to_monthly_last(df: pd.DataFrame) -> pd.DataFrame:
    # Use 'M' (month-end) for maximum pandas compatibility
    out = df.resample("M").last().dropna()
    if out.empty:
        raise RuntimeError("Series became empty after monthly resample.")
    return out


def yoy_percent(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["value"] = out["value"].pct_change(12) * 100.0
    out = out.dropna()
    if out.empty:
        raise RuntimeError("Series became empty after YoY transform.")
    return out


def delta_12m_bps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["value"] = (out["value"] - out["value"].shift(12)) * 100.0
    out = out.dropna()
    if out.empty:
        raise RuntimeError("Series became empty after 12m bps transform.")
    return out


def compute_10y_zscore(df: pd.DataFrame):
    latest_date = df.index[-1]
    latest_value = float(df["value"].iloc[-1])

    cutoff_date = latest_date - pd.DateOffset(years=10)
    ten_years = df.loc[df.index >= cutoff_date]

    if len(ten_years) < 24:
        raise RuntimeError("Not enough data in the last 10 years to compute stable z-score.")

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
    subtitle: str,
    state: str,
    outfile: str,
    value_suffix: str = ""
) -> None:
    v = max(min_val, min(value, max_val))
    frac = (v - min_val) / (max_val - min_val) if max_val > min_val else 0.0
    angle = math.pi * (1 - frac)

    fig, ax = plt.subplots(figsize=(7, 4))

    theta = np.linspace(0, math.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), linewidth=10, color=ARC_COLOR)

    ticks = 6
    for i in range(ticks + 1):
        tfrac = i / ticks
        tang = math.pi * (1 - tfrac)
        x0, y0 = 0.92 * math.cos(tang), 0.92 * math.sin(tang)
        x1, y1 = 1.02 * math.cos(tang), 1.02 * math.sin(tang)
        ax.plot([x0, x1], [y0, y1], linewidth=2, color="black")

        tval = min_val + tfrac * (max_val - min_val)
        ax.text(
            1.15 * math.cos(tang),
            1.15 * math.sin(tang),
            f"{tval:.0f}{value_suffix}",
            ha="center",
            va="center",
            fontsize=10,
            color="black",
        )

    ax.plot([0, 0.85 * math.cos(angle)], [0, 0.85 * math.sin(angle)],
            linewidth=6, color=NEEDLE_COLOR)
    ax.scatter([0], [0], s=250, color=CENTER_COLOR)

    ax.text(0, 1.25, title, ha="center", fontsize=16, fontweight="bold", color="black")
    ax.text(0, -0.25, f"{v:.2f}{value_suffix}", ha="center", fontsize=24, fontweight="bold", color="black")
    ax.text(0, -0.45, f"Regime: {state}", ha="center", fontsize=14, fontweight="bold", color=NEEDLE_COLOR)
    ax.text(0, -0.65, subtitle, ha="center", fontsize=11, color="black")

    ax.set_aspect("equal")
    ax.axis("off")

    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def inflation_regime_from_z(z: float) -> str:
    return "Cooling" if z < -0.5 else ("Sticky" if z < 0.5 else "Rising")


def credit_regime_from_z(z: float) -> str:
    return "Calm" if z < -0.5 else ("Cautious" if z < 0.5 else "Stressed")


def rates_regime_from_z(z: float) -> str:
    return "Supportive" if z < -0.5 else ("Neutral" if z < 0.5 else "Restrictive")


DIAL_CONFIG = [
    {
        "series_id": "CPIAUCSL",
        "fred_units": "pc1",
        "dial_min": 0.0,
        "dial_max": 6.0,
        "title": "Inflation Backdrop",
        "suffix": "%",
        "regime_fn": inflation_regime_from_z,
        "outfile": "inflation_dial.png",
    },
    {
        "series_id": "DGS10",
        "fred_units": "lin",
        "dial_min": 0.0,
        "dial_max": 6.0,
        "title": "Rates Backdrop (10Y Treasury)",
        "suffix": "%",
        "regime_fn": rates_regime_from_z,
        "outfile": "rates_dial.png",
    },
    {
        "series_id": "BAMLH0A0HYM2",
        "fred_units": "lin",
        "dial_min": 2.0,
        "dial_max": 10.0,
        "title": "Credit Spreads (HY OAS)",
        "suffix": "%",
        "regime_fn": credit_regime_from_z,
        "outfile": "credit_spreads_dial.png",
    },
]

DASHBOARD_CONFIG = [
    {"label": "Effective Fed Funds Rate (%)", "series_id": "EFFR", "transform": "level"},
    {"label": "Yield Curve: 10Y - 3M (pp)", "series_id": "T10Y3M", "transform": "level"},
    {"label": "30Y Mortgage Rate (%)", "series_id": "MORTGAGE30US", "transform": "level"},
    {"label": "M2 (YoY %)", "series_id": "M2SL", "transform": "yoy"},
    {"label": "Core PCE (YoY %)", "series_id": "PCEPILFE", "transform": "yoy"},
    {"label": "CPI (YoY %)", "series_id": "CPIAUCSL", "transform": "yoy"},
    {"label": "Housing Permits (YoY %)", "series_id": "PERMIT", "transform": "yoy"},
    {"label": "Initial Claims (Level)", "series_id": "ICSA", "transform": "level"},
    {"label": "Trade-Weighted USD (Broad)", "series_id": "DTWEXBGS", "transform": "level"},
    {"label": "Global EPU Index", "series_id": "GEPUCURRENT", "transform": "level"},
]


def apply_transform(df: pd.DataFrame, transform: str) -> pd.DataFrame:
    if transform == "level":
        return df
    if transform == "yoy":
        return yoy_percent(df)
    if transform == "delta_12m_bps":
        return delta_12m_bps(df)
    raise ValueError(f"Unknown transform: {transform}")


def plot_dashboard(series_map: dict, outfile: str, months: int = 120) -> None:
    n = len(series_map)
    cols = 2
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(14, 4.2 * rows))
    for i, (title, df) in enumerate(series_map.items(), start=1):
        ax = fig.add_subplot(rows, cols, i)
        tail = df.tail(months)
        ax.plot(tail.index, tail["value"])
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    api_key = os.getenv(FRED_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{FRED_API_KEY_ENV} not found. Set it and restart your terminal.")

    outputs = []

    print("Building dials...")
    for cfg in DIAL_CONFIG:
        sid = cfg["series_id"]
        try:
            print(f"  Pulling {sid}...")
            df = fred_series_observations(sid, api_key=api_key, units=cfg["fred_units"])
            df_m = resample_to_monthly_last(df)
            latest, mean_10y, std_10y, z, dt = compute_10y_zscore(df_m)
            state = cfg["regime_fn"](z)

            outpath = os.path.abspath(os.path.join(OUTPUT_DIR, cfg["outfile"]))
            draw_dial(
                latest,
                min_val=cfg["dial_min"],
                max_val=cfg["dial_max"],
                title=cfg["title"],
                subtitle=f"As of {dt.date()} | Z = {z:.2f}",
                state=state,
                outfile=outpath,
                value_suffix=cfg["suffix"],
            )
            print(f"    Saved: {outpath}")
            outputs.append(outpath)
        except Exception as e:
            print(f"    FAILED on {sid}: {e}")

    print("Building dashboard panels...")
    panels = {}
    for item in DASHBOARD_CONFIG:
        sid = item["series_id"]
        try:
            print(f"  Pulling {sid} ({item['label']})...")
            df = fred_series_observations(sid, api_key=api_key, units="lin")
            df_m = resample_to_monthly_last(df)
            df_t = apply_transform(df_m, item["transform"])
            panels[item["label"]] = df_t
        except Exception as e:
            print(f"    FAILED on {sid}: {e}")

    if panels:
        dash_out = os.path.abspath(os.path.join(OUTPUT_DIR, "macro_dashboard.png"))
        plot_dashboard(panels, dash_out, months=120)
        print(f"Saved dashboard: {dash_out}")
        outputs.append(dash_out)
    else:
        print("No panels were created. Something is very wrong (likely API key or connectivity).")

    try:
        for p in outputs:
            os.startfile(p)
    except Exception as e:
        print(f"Could not auto-open outputs: {e}")


if __name__ == "__main__":
    main()
