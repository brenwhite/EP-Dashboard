from __future__ import annotations

from datetime import date
from html import escape
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


st.set_page_config(page_title="Institutional Market Overview", page_icon=":bar_chart:", layout="wide")


CSV_FILENAME = "data.csv"
CLASS_COL = "Classification"
CURATED_CLASS_COL = "Asset Class"

W_TREND = 0.40
W_MOM = 0.30
W_RISK = 0.20
W_PART = 0.10

P_W = 0.40
P_M = 0.35
P_Y = 0.25

EQUITY_CURATED_CLASS = "Equity Markets"
FRED_API_KEY_ENV = "FRED_API_KEY"
FACTORS_API_BASE = "https://factorstoday.com"
FACTORS_API_BASE_CANDIDATES = [
    "https://factorstoday.com",
    "https://www.factorstoday.com",
]

MACRO_DIAL_CONFIG = [
    {
        "series_id": "CPIAUCSL",
        "fred_units": "pc1",
        "title": "Inflation Backdrop",
        "suffix": "%",
        "regime_labels": ("Cooling", "Sticky", "Rising"),
        "description": "CPI year-over-year",
    },
    {
        "series_id": "DGS10",
        "fred_units": "lin",
        "title": "Rates Backdrop",
        "suffix": "%",
        "regime_labels": ("Supportive", "Neutral", "Restrictive"),
        "description": "10Y Treasury yield",
    },
    {
        "series_id": "BAMLH0A0HYM2",
        "fred_units": "lin",
        "title": "Credit Spreads",
        "suffix": "%",
        "regime_labels": ("Calm", "Cautious", "Stressed"),
        "description": "High-yield OAS",
    },
]

BASIC_STATE_FACTORS = [
    "OilPrice",
    "InterestRate",
    "Buybacks",
    "BetaFactor",
    "Momentum",
    "USDollar",
    "SmallSize",
    "CreditRisk",
    "MarketBreadth",
    "Growth",
    "DividendYield",
    "Value",
    "GoldPrice",
    "Market",
    "Quality",
    "LowVolatility",
    "Liquidity",
]

CURATED_TEMPLATE_ROWS = [
    {"Ticker": "SPY", "Display_Name": "S&P 500", CURATED_CLASS_COL: "Equity Markets"},
    {"Ticker": "SCHF", "Display_Name": "International Developed", CURATED_CLASS_COL: "Equity Markets"},
    {"Ticker": "EZU", "Display_Name": "Eurozone Markets", CURATED_CLASS_COL: "Equity Markets"},
    {"Ticker": "EWJ", "Display_Name": "Japan", CURATED_CLASS_COL: "Equity Markets"},
    {"Ticker": "FXI", "Display_Name": "Large Cap China", CURATED_CLASS_COL: "Equity Markets"},
    {"Ticker": "EEM", "Display_Name": "Emerging Markets", CURATED_CLASS_COL: "Equity Markets"},
    {"Ticker": "USO", "Display_Name": "Brent Crude", CURATED_CLASS_COL: "Commodities"},
    {"Ticker": "GSG", "Display_Name": "S&P GSCI Commodity", CURATED_CLASS_COL: "Commodities"},
    {"Ticker": "GLD", "Display_Name": "Gold", CURATED_CLASS_COL: "Commodities"},
    {"Ticker": "SLV", "Display_Name": "Silver", CURATED_CLASS_COL: "Commodities"},
    {"Ticker": "SHV", "Display_Name": "U.S. T-Bills", CURATED_CLASS_COL: "U.S. Government Debt"},
    {"Ticker": "IEF", "Display_Name": "U.S. Intermediate Term Treasuries", CURATED_CLASS_COL: "U.S. Government Debt"},
    {"Ticker": "TLT", "Display_Name": "U.S. Long Term Treasuries", CURATED_CLASS_COL: "U.S. Government Debt"},
    {"Ticker": "TIP", "Display_Name": "Intermediate TIPS Real Return", CURATED_CLASS_COL: "U.S. Government Debt"},
    {"Ticker": "AGG", "Display_Name": "U.S. Aggregate Bond Market", CURATED_CLASS_COL: "U.S. Corp Credit"},
    {"Ticker": "BKLN", "Display_Name": "Senior Loans", CURATED_CLASS_COL: "U.S. Corp Credit"},
    {"Ticker": "LQD", "Display_Name": "U.S. Corporate Bond Market", CURATED_CLASS_COL: "U.S. Corp Credit"},
    {"Ticker": "HYG", "Display_Name": "U.S. High Yield Bond Market", CURATED_CLASS_COL: "U.S. Corp Credit"},
    {"Ticker": "BNDX", "Display_Name": "Total International Bond Market", CURATED_CLASS_COL: "International Credit"},
    {"Ticker": "EMB", "Display_Name": "Emerging Market Bond Market", CURATED_CLASS_COL: "International Credit"},
]


def linear_score(value, low: float, high: float):
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
    if pd.isna(price) or pd.isna(ema) or ema == 0:
        return np.nan
    dist = (price - ema) / ema
    return float(np.clip(((dist - low) / (high - low)) * 100.0, 0.0, 100.0))


def score_rsi(val: float) -> float:
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


def safe_pct_series(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    med = series.abs().median(skipna=True)
    if pd.notna(med) and med > 1:
        return series / 100.0
    return series


def flow_percentile_scores(df: pd.DataFrame, flow_col: str, tickers: list[str]) -> pd.Series:
    subset = df[df["Ticker"].isin(tickers)].copy()
    flows = pd.to_numeric(subset[flow_col], errors="coerce")
    if flows.notna().sum() == 0:
        return pd.Series(index=df.index, data=np.nan, dtype=float)
    pct = flows.rank(pct=True) * 100.0
    output = pd.Series(index=df.index, data=np.nan, dtype=float)
    output.loc[subset.index] = pct.values
    return output


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


def pick_col(df: pd.DataFrame, preferred: str, fallback: str) -> str:
    if preferred in df.columns:
        return preferred
    if fallback in df.columns:
        return fallback
    raise ValueError(f"Missing required column: '{preferred}' or '{fallback}'")


def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Missing required column. Expected one of: {candidates}")


def parse_pe_string(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower().replace("x", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def first_valid(*values):
    for value in values:
        if value is not None and not pd.isna(value):
            return value
    return None


def format_percent(value: float | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "&mdash;"
    return f"{value:.{decimals}f}%"


def format_percent_from_decimal(value: float | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "&mdash;"
    return f"{value * 100:.{decimals}f}%"


def format_pe_value(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "&mdash;"
    return f"{value:.1f}x"


def format_pe_pair(trailing: float | None, forward: float | None) -> str:
    return f"{format_pe_value(trailing)} / {format_pe_value(forward)}"


def render_master_bar(score: float) -> str:
    value = 0.0 if pd.isna(score) else float(np.clip(score, 0, 100))
    return (
        '<div class="score-wrap">'
        f'<div class="score-bar"><span style="width:{value:.1f}%"></span></div>'
        f'<div class="score-label">{value:.1f}</div>'
        "</div>"
    )


def regime_indicator(regime: str) -> str:
    if regime in {"Strong Bull", "Bull"}:
        return '<span class="signal bull">&#8593;</span>'
    if regime in {"Strong Bear", "Bear"}:
        return '<span class="signal bear">&#8595;</span>'
    return '<span class="signal neutral">&#8599;</span>'


def extract_info_value(payload: object, keys: list[str]) -> float | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if value is not None and not pd.isna(value):
                return value
        return None
    for key in keys:
        if hasattr(payload, key):
            value = getattr(payload, key)
            if value is not None and not pd.isna(value):
                return value
    for attr in ("_asdict", "to_dict"):
        if hasattr(payload, attr):
            try:
                nested = getattr(payload, attr)()
            except TypeError:
                continue
            value = extract_info_value(nested, keys)
            if value is not None:
                return value
    return None


def normalize_yield_to_percent(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value * 100) if abs(value) <= 1 else float(value)


def get_fred_api_key() -> str | None:
    try:
        secret_val = st.secrets.get(FRED_API_KEY_ENV)
        if secret_val:
            return str(secret_val)
    except Exception:
        pass
    return os.getenv(FRED_API_KEY_ENV)


def fred_series_observations(series_id: str, api_key: str, units: str = "lin") -> pd.DataFrame:
    response = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "api_key": api_key,
            "file_type": "json",
            "series_id": series_id,
            "units": units,
            "sort_order": "asc",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    observations = payload.get("observations", [])
    if not observations:
        raise RuntimeError(f"No observations returned for series {series_id}")

    df = pd.DataFrame(observations)[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().set_index("date").sort_index()
    if df.empty:
        raise RuntimeError(f"Series {series_id} became empty after cleaning.")
    return df


def resample_to_monthly_last(df: pd.DataFrame) -> pd.DataFrame:
    out = df.resample("M").last().dropna()
    if out.empty:
        raise RuntimeError("Series became empty after monthly resample.")
    return out


def compute_10y_zscore(df: pd.DataFrame) -> tuple[float, float, float, float, pd.Timestamp]:
    latest_date = df.index[-1]
    latest_value = float(df["value"].iloc[-1])
    cutoff_date = latest_date - pd.DateOffset(years=10)
    ten_years = df.loc[df.index >= cutoff_date]
    if len(ten_years) < 24:
        raise RuntimeError("Not enough data in the last 10 years to compute a stable z-score.")
    mean_10y = float(ten_years["value"].mean())
    std_10y = float(ten_years["value"].std())
    if std_10y == 0 or np.isnan(std_10y):
        raise RuntimeError("10Y std dev invalid.")
    z = (latest_value - mean_10y) / std_10y
    return latest_value, mean_10y, std_10y, z, latest_date


def macro_regime_from_z(z_score: float, labels: tuple[str, str, str]) -> str:
    if z_score < -0.5:
        return labels[0]
    if z_score < 0.5:
        return labels[1]
    return labels[2]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_macro_backdrop() -> pd.DataFrame:
    api_key = get_fred_api_key()
    rows: list[dict[str, object]] = []
    if not api_key:
        return pd.DataFrame(rows)

    for cfg in MACRO_DIAL_CONFIG:
        try:
            series = fred_series_observations(cfg["series_id"], api_key=api_key, units=cfg["fred_units"])
            series_m = resample_to_monthly_last(series)
            latest, mean_10y, std_10y, z_score, latest_date = compute_10y_zscore(series_m)
            rows.append(
                {
                    "Title": cfg["title"],
                    "Series_ID": cfg["series_id"],
                    "Description": cfg["description"],
                    "Latest": latest,
                    "Mean_10Y": mean_10y,
                    "Std_10Y": std_10y,
                    "Z_Score": z_score,
                    "As_Of": latest_date,
                    "Suffix": cfg["suffix"],
                    "Regime": macro_regime_from_z(z_score, cfg["regime_labels"]),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "Title": cfg["title"],
                    "Series_ID": cfg["series_id"],
                    "Description": cfg["description"],
                    "Latest": np.nan,
                    "Mean_10Y": np.nan,
                    "Std_10Y": np.nan,
                    "Z_Score": np.nan,
                    "As_Of": pd.NaT,
                    "Suffix": cfg["suffix"],
                    "Regime": f"Unavailable: {exc}",
                }
            )

    return pd.DataFrame(rows)


def factor_api_get_json(path: str, timeout: int = 30) -> dict:
    last_error: Exception | None = None
    for base_url in FACTORS_API_BASE_CANDIDATES:
        try:
            response = requests.get(f"{base_url}{path}", timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.SSLError as exc:
            last_error = exc
            continue
        except requests.RequestException as exc:
            last_error = exc
            continue
    if last_error is None:
        raise RuntimeError(f"Unable to fetch factor API path: {path}")
    raise RuntimeError(f"Factor API request failed for {path}: {last_error}")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_factor_historic_snapshot() -> pd.DataFrame:
    payload = factor_api_get_json("/api/factor-returns/historic", timeout=30)

    rows: list[dict[str, float | str | None]] = []
    for factor_name, horizons in payload.items():
        row: dict[str, float | str | None] = {"Factor": factor_name}
        for horizon in ("21d", "63d", "126d", "252d"):
            cell = horizons.get(horizon, {}) if isinstance(horizons, dict) else {}
            row[f"{horizon}_value"] = cell.get("value")
            row[f"{horizon}_z"] = cell.get("zScore")
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_factor_daily_return_panel() -> pd.DataFrame:
    payload = factor_api_get_json("/api/factors/download-all?format=json", timeout=60)
    data = payload.get("data", [])
    if not data:
        raise RuntimeError("Factor daily returns payload was empty.")

    df = pd.DataFrame(data)
    if "date" not in df.columns:
        raise RuntimeError("Factor daily returns payload missing 'date'.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def compute_rsi(series: pd.Series, window: int) -> float | None:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    latest = rsi.iloc[-1]
    return None if pd.isna(latest) else float(latest)


def compute_factor_signal_frame(return_panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | None]] = []
    for factor in return_panel.columns:
        daily_returns = return_panel[factor].dropna()
        if len(daily_returns) < 260:
            continue

        close = (1.0 + daily_returns).cumprod() * 100.0
        if close.empty:
            continue

        last_price = float(close.iloc[-1])
        ema5 = float(close.ewm(span=5, adjust=False).mean().iloc[-1])
        ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])

        rsi14 = compute_rsi(close, 14)
        rsi10 = compute_rsi(close, 10)

        def trailing_return(period: int) -> float | None:
            if len(close) <= period:
                return None
            start = close.iloc[-(period + 1)]
            end = close.iloc[-1]
            if pd.isna(start) or start == 0:
                return None
            return float((end / start) - 1)

        def trailing_vol(period: int) -> float | None:
            if len(daily_returns) < period:
                return None
            window = daily_returns.tail(period)
            vol = float(window.std() * np.sqrt(252))
            return None if pd.isna(vol) else vol

        ret_1w = trailing_return(5)
        ret_1m = trailing_return(21)
        ret_3m = trailing_return(63)
        ret_6m = trailing_return(126)
        ret_1y = trailing_return(252)

        vol_1m = trailing_vol(21)
        vol_3m = trailing_vol(63)
        vol_6m = trailing_vol(126)
        vol_1y = trailing_vol(252)

        rolling_high = close.tail(252).max()
        below_52w = None if pd.isna(rolling_high) or rolling_high == 0 else float((last_price - rolling_high) / rolling_high)

        price_above_score = (
            float(last_price > ema5) +
            float(last_price > ema21) +
            float(last_price > ema50) +
            float(last_price > ema200)
        ) / 4.0 * 100.0

        ema_order_score = (
            float(ema5 > ema21) +
            float(ema21 > ema50) +
            float(ema50 > ema200)
        ) / 3.0 * 100.0

        stack_score = 0.60 * price_above_score + 0.40 * ema_order_score
        dist_long_score = np.nanmean([dist_score(last_price, ema50), dist_score(last_price, ema200)])
        trend_score = 0.50 * stack_score + 0.50 * dist_long_score

        cross_score = 100.0 if ema5 > ema21 else 0.0
        p1w = linear_score(ret_1w, low=-0.05, high=0.05)
        p1m = linear_score(ret_1m, low=-0.10, high=0.10)
        p3m = linear_score(ret_3m, low=-0.15, high=0.15)
        p6m = linear_score(ret_6m, low=-0.25, high=0.25)
        p1y = linear_score(ret_1y, low=-0.40, high=0.40)
        price_mom_score = 0.20 * p1w + 0.25 * p1m + 0.25 * p3m + 0.20 * p6m + 0.10 * p1y
        momentum_score = (
            0.35 * score_rsi(rsi14) +
            0.25 * score_rsi(rsi10) +
            0.15 * cross_score +
            0.25 * price_mom_score
        )
        return_strength = 0.70 * trend_score + 0.30 * momentum_score

        rows.append(
            {
                "Factor": factor,
                "Last Price": last_price,
                "EMA (5D)": ema5,
                "EMA (21D)": ema21,
                "EMA (50D)": ema50,
                "EMA (200D)": ema200,
                "RSI": rsi14,
                "RSI 10D": rsi10,
                "Price Chg. % (1W)": ret_1w,
                "Price Chg. % (1M)": ret_1m,
                "Price Chg. % (3M)": ret_3m,
                "Price Chg. % (6M)": ret_6m,
                "Price Chg. % (1Y)": ret_1y,
                "Volatility (1M)": vol_1m,
                "Volatility (3M)": vol_3m,
                "Volatility (6M)": vol_6m,
                "Volatility (1Y)": vol_1y,
                "Below 52W High %": below_52w,
                "Trend_Score": trend_score,
                "Momentum_Score": momentum_score,
                "Return_Strength": return_strength,
            }
        )

    return pd.DataFrame(rows)


def get_close_frame(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    if isinstance(history.columns, pd.MultiIndex):
        if "Close" not in history.columns.get_level_values(0):
            return pd.DataFrame()
        close = history["Close"].copy()
        if isinstance(close, pd.Series):
            close = close.to_frame()
        return close
    if "Close" in history.columns:
        return history[["Close"]].rename(columns={"Close": "SINGLE"})
    return pd.DataFrame()


def calculate_period_return(closes: pd.Series, anchor_date: pd.Timestamp) -> float | None:
    closes = closes.dropna()
    if closes.empty:
        return None
    eligible = closes[closes.index >= anchor_date]
    if eligible.empty:
        return None
    start_price = eligible.iloc[0]
    end_price = closes.iloc[-1]
    if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
        return None
    return float((end_price / start_price) - 1)


def compute_live_return_maps(tickers: list[str]) -> dict[str, dict[str, float | None]]:
    horizons = ("YTD_Return", "Return_1Y", "Return_3Y", "Return_5Y")
    empty = {ticker: {h: None for h in horizons} for ticker in tickers}
    if yf is None or not tickers:
        return empty

    today = pd.Timestamp.today().normalize()
    start = date(today.year - 6, 1, 1)
    try:
        history = yf.download(
            tickers=tickers,
            start=start,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
    except Exception:
        return empty

    close_frame = get_close_frame(history)
    if close_frame.empty:
        return empty

    anchor_map = {
        "YTD_Return": pd.Timestamp(date(today.year, 1, 1)),
        "Return_1Y": today - pd.DateOffset(years=1),
        "Return_3Y": today - pd.DateOffset(years=3),
        "Return_5Y": today - pd.DateOffset(years=5),
    }

    results = empty.copy()
    if list(close_frame.columns) == ["SINGLE"] and len(tickers) == 1:
        single_closes = close_frame["SINGLE"]
        results[tickers[0]] = {
            horizon: calculate_period_return(single_closes, anchor)
            for horizon, anchor in anchor_map.items()
        }
        return results

    for ticker in tickers:
        if ticker not in close_frame.columns:
            continue
        ticker_closes = close_frame[ticker]
        results[ticker] = {
            horizon: calculate_period_return(ticker_closes, anchor)
            for horizon, anchor in anchor_map.items()
        }
    return results


@st.cache_data(show_spinner=False)
def load_market_data(csv_name: str) -> pd.DataFrame:
    csv_path = Path(__file__).resolve().parent / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    for column in df.columns:
        if column not in {"Ticker", "Name", CLASS_COL, "Inception Date", "Class"}:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    pct_cols = [
        "Below 52W High %",
        "Price Chg. % (1W)",
        "Price Chg. % (1M)",
        "Price Chg. % (3M)",
        "Price Chg. % (6M)",
        "Price Chg. % (1Y)",
        "Price Chg. % (3Y)",
        "Price Chg. % (5Y)",
        "Price Chg. % (10Y)",
    ]
    for column in pct_cols:
        if column in df.columns:
            df[column] = safe_pct_series(df[column])
    return df


@st.cache_data(show_spinner=False)
def load_curated_template() -> pd.DataFrame:
    assets = pd.DataFrame(CURATED_TEMPLATE_ROWS).copy()
    assets["Template_Order"] = np.arange(len(assets))
    assets["Template_Yield"] = np.nan
    assets["Template_YTD"] = np.nan
    assets["Template_1Y"] = np.nan
    assets["Template_3Y"] = np.nan
    assets["Template_5Y"] = np.nan
    assets["Template_Below52"] = np.nan
    assets["Template_PE_Trailing"] = np.nan
    assets["Template_PE_Forward"] = np.nan
    return assets[
        [
            "Ticker",
            "Display_Name",
            CURATED_CLASS_COL,
            "Template_Order",
            "Template_Yield",
            "Template_YTD",
            "Template_1Y",
            "Template_3Y",
            "Template_5Y",
            "Template_Below52",
            "Template_PE_Trailing",
            "Template_PE_Forward",
        ]
    ]


def compute_market_scores(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    tickers = sorted(data["Ticker"].dropna().unique().tolist())
    ema_short = pick_col(data, "EMA (21D)", "EMA (20D)")
    optional_emas = [col for col in ["EMA (100D)", "EMA (150D)", "EMA (250D)", "EMA (300D)"] if col in data.columns]
    ema_list = ["EMA (5D)", ema_short, "EMA (50D)"] + optional_emas + ["EMA (200D)"]
    if "EMA (200D)" in ema_list:
        ema_list = [ema for ema in ema_list if ema != "EMA (200D)"] + ["EMA (200D)"]
    for extra in ["EMA (250D)", "EMA (300D)"]:
        if extra in data.columns and extra not in ema_list:
            ema_list.append(extra)

    above_votes = [(data["Last Price"] > data[ema]).astype(float) for ema in ema_list]
    price_above_score = (sum(above_votes) / len(above_votes)) * 100.0
    order_votes = [(data[a] > data[b]).astype(float) for a, b in zip(ema_list[:-1], ema_list[1:])]
    ema_order_score = (sum(order_votes) / len(order_votes)) * 100.0 if order_votes else 50.0
    stack_score = 0.60 * price_above_score + 0.40 * ema_order_score

    d50 = data.apply(lambda row: dist_score(row["Last Price"], row["EMA (50D)"]), axis=1)
    d200 = data.apply(lambda row: dist_score(row["Last Price"], row["EMA (200D)"]), axis=1)
    dist_components = [d50, d200]
    if "EMA (300D)" in data.columns:
        d300 = data.apply(lambda row: dist_score(row["Last Price"], row["EMA (300D)"]), axis=1)
        dist_components.append(d300)
    dist_long_score = pd.concat(dist_components, axis=1).mean(axis=1, skipna=True)
    data["Trend_Score"] = 0.50 * stack_score + 0.50 * dist_long_score

    rsi_score = data["RSI"].apply(score_rsi)
    rsi10_score = data["RSI 10D"].apply(score_rsi)
    cross_score = np.where(data["EMA (5D)"] > data[ema_short], 100.0, 0.0)
    p1w = linear_score(data["Price Chg. % (1W)"], low=-0.05, high=0.05)
    p1m = linear_score(data["Price Chg. % (1M)"], low=-0.10, high=0.10)
    p3m = linear_score(data["Price Chg. % (3M)"], low=-0.15, high=0.15)
    p6m = linear_score(data["Price Chg. % (6M)"], low=-0.25, high=0.25)
    p1y = linear_score(data["Price Chg. % (1Y)"], low=-0.40, high=0.40)
    price_mom_score = 0.20 * p1w + 0.25 * p1m + 0.25 * p3m + 0.20 * p6m + 0.10 * p1y
    data["Momentum_Score"] = 0.35 * rsi_score + 0.25 * rsi10_score + 0.15 * cross_score + 0.25 * price_mom_score

    v1 = data["Volatility (1M)"]
    v3 = data["Volatility (3M)"]
    v6 = data["Volatility (6M)"]
    vy = data["Volatility (1Y)"]
    worse_count = (v1 > vy).astype(float) + (v3 > vy).astype(float) + (v6 > vy).astype(float)
    vol_regime_score = worse_count.map({0.0: 90.0, 1.0: 70.0, 2.0: 45.0, 3.0: 20.0})
    drawdown_score = linear_score(data["Below 52W High %"], low=-0.20, high=0.0)
    data["Risk_Score"] = 0.70 * vol_regime_score + 0.30 * drawdown_score

    part_w = flow_percentile_scores(data, "Fund Flows/Periodic (W)", tickers)
    part_m = flow_percentile_scores(data, "Fund Flows/Periodic (M)", tickers)
    part_y = flow_percentile_scores(data, "Fund Flows/Periodic (Y)", tickers)
    data["Participation_Score"] = P_W * part_w + P_M * part_m + P_Y * part_y

    data["Master_Score"] = (
        W_TREND * data["Trend_Score"]
        + W_MOM * data["Momentum_Score"]
        + W_RISK * data["Risk_Score"]
        + W_PART * data["Participation_Score"]
    )
    data["Regime"] = data["Master_Score"].apply(regime_from_score)
    return data


def compute_classification_scores(df_scored: pd.DataFrame) -> pd.DataFrame:
    if CLASS_COL not in df_scored.columns:
        raise ValueError(f"Missing '{CLASS_COL}' column.")

    grouped = df_scored.groupby(CLASS_COL, dropna=False)
    output = grouped[["Trend_Score", "Momentum_Score", "Risk_Score", "Participation_Score", "Master_Score"]].mean()
    output["Count"] = grouped.size()
    output["Master_Score"] = (
        W_TREND * output["Trend_Score"]
        + W_MOM * output["Momentum_Score"]
        + W_RISK * output["Risk_Score"]
        + W_PART * output["Participation_Score"]
    )
    output = output.reset_index().rename(columns={CLASS_COL: "Classification"})
    output["Regime"] = output["Master_Score"].apply(regime_from_score)
    return output.sort_values("Master_Score", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yfinance_enrichment(tickers: list[str]) -> pd.DataFrame:
    columns = ["Ticker", "Yield", "PE_Trailing", "PE_Forward", "YTD_Return", "Return_1Y", "Return_3Y", "Return_5Y"]
    if not tickers:
        return pd.DataFrame(columns=columns)
    if yf is None:
        return pd.DataFrame(
            {
                "Ticker": tickers,
                "Yield": [None] * len(tickers),
                "PE_Trailing": [None] * len(tickers),
                "PE_Forward": [None] * len(tickers),
                "YTD_Return": [None] * len(tickers),
                "Return_1Y": [None] * len(tickers),
                "Return_3Y": [None] * len(tickers),
                "Return_5Y": [None] * len(tickers),
            }
        )

    return_map = compute_live_return_maps(tickers)
    rows: list[dict[str, float | str | None]] = []
    yield_keys = ["secYield", "secYieldPct", "yield", "yieldPct", "trailingAnnualDividendYield", "dividendYield"]
    pe_trailing_keys = ["trailingPE", "trailingPe", "peRatio"]
    pe_forward_keys = ["forwardPE", "forwardPe", "forwardPE1Yr"]
    price_keys = ["currentPrice", "regularMarketPrice", "lastPrice"]
    forward_eps_keys = ["forwardEps", "forwardEpsCurrentYear"]

    for ticker in tickers:
        sec_yield = None
        trailing_pe = None
        forward_pe = None
        ticker_obj = None
        basic_info = {}
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.get_info()
        except Exception:
            try:
                info = ticker_obj.info if ticker_obj is not None else {}
            except Exception:
                info = {}
        try:
            funds_data = getattr(ticker_obj, "funds_data", None)
        except Exception:
            funds_data = None

        sec_yield = extract_info_value(info, yield_keys)
        if sec_yield is None:
            sec_yield = extract_info_value(funds_data, yield_keys)
        trailing_pe = extract_info_value(info, pe_trailing_keys)
        if trailing_pe is None:
            trailing_pe = extract_info_value(funds_data, pe_trailing_keys)
        forward_pe = extract_info_value(info, pe_forward_keys)
        if forward_pe is None:
            forward_pe = extract_info_value(funds_data, pe_forward_keys)
        if forward_pe is None:
            try:
                basic_info = ticker_obj.basic_info if ticker_obj is not None else {}
            except Exception:
                basic_info = {}
            forward_pe = extract_info_value(basic_info, pe_forward_keys)
        if forward_pe is None:
            current_price = first_valid(
                extract_info_value(info, price_keys),
                extract_info_value(basic_info, price_keys),
            )
            forward_eps = first_valid(
                extract_info_value(info, forward_eps_keys),
                extract_info_value(basic_info, forward_eps_keys),
            )
            if current_price is not None and forward_eps not in (None, 0) and not pd.isna(forward_eps):
                forward_pe = float(current_price) / float(forward_eps)
        if forward_pe is not None and trailing_pe is not None and forward_pe > 1000:
            forward_pe = None

        rows.append(
            {
                "Ticker": ticker,
                "Yield": normalize_yield_to_percent(sec_yield),
                "PE_Trailing": None if trailing_pe is None or pd.isna(trailing_pe) else float(trailing_pe),
                "PE_Forward": None if forward_pe is None or pd.isna(forward_pe) else float(forward_pe),
                "YTD_Return": return_map.get(ticker, {}).get("YTD_Return"),
                "Return_1Y": return_map.get(ticker, {}).get("Return_1Y"),
                "Return_3Y": return_map.get(ticker, {}).get("Return_3Y"),
                "Return_5Y": return_map.get(ticker, {}).get("Return_5Y"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_universe_frame(df: pd.DataFrame) -> pd.DataFrame:
    view = df.copy()
    view["Yield_Display"] = view["Yield"].apply(lambda value: format_percent(value, decimals=2))
    view["YTD_Display"] = view["YTD_Return"].apply(format_percent_from_decimal)
    view["Return_1Y_Display"] = view["Return_1Y"].apply(format_percent_from_decimal)
    view["Return_3Y_Display"] = view["Return_3Y"].apply(format_percent_from_decimal)
    view["Return_5Y_Display"] = view["Return_5Y"].apply(format_percent_from_decimal)
    view["PE_Display"] = view.apply(lambda row: format_pe_pair(row["PE_Trailing"], row["PE_Forward"]), axis=1)
    return view


def build_curated_frame(scored_df: pd.DataFrame, enrichment_df: pd.DataFrame, template_df: pd.DataFrame) -> pd.DataFrame:
    merged = template_df.merge(scored_df, on="Ticker", how="left", suffixes=("", "_csv"))
    merged = merged.merge(enrichment_df, on="Ticker", how="left")

    merged["Name_Final"] = merged["Display_Name"].combine_first(merged["Name"])
    merged["Yield_Final"] = merged.apply(lambda row: first_valid(row["Yield"], row["Template_Yield"]), axis=1)
    merged["YTD_Final"] = merged.apply(lambda row: first_valid(row["YTD_Return"], row["Template_YTD"]), axis=1)
    merged["Return_1Y_Final"] = merged["Return_1Y"]
    merged["Return_3Y_Final"] = merged["Return_3Y"]
    merged["Return_5Y_Final"] = merged["Return_5Y"]
    merged["Below52_Final"] = merged.apply(lambda row: first_valid(row["Below 52W High %"], row["Template_Below52"]), axis=1)
    merged["PE_Trailing_Final"] = merged.apply(lambda row: first_valid(row["PE_Trailing"], row["Template_PE_Trailing"]), axis=1)
    merged["PE_Forward_Final"] = merged.apply(lambda row: first_valid(row["PE_Forward"], row["Template_PE_Forward"]), axis=1)

    non_equity_mask = merged[CURATED_CLASS_COL] != EQUITY_CURATED_CLASS
    merged.loc[non_equity_mask, "PE_Trailing_Final"] = np.nan
    merged.loc[non_equity_mask, "PE_Forward_Final"] = np.nan

    merged["Yield_Display"] = merged["Yield_Final"].apply(lambda value: format_percent(value, decimals=2))
    merged["YTD_Display"] = merged["YTD_Final"].apply(format_percent_from_decimal)
    merged["Return_1Y_Display"] = merged["Return_1Y_Final"].apply(format_percent_from_decimal)
    merged["Return_3Y_Display"] = merged["Return_3Y_Final"].apply(format_percent_from_decimal)
    merged["Return_5Y_Display"] = merged["Return_5Y_Final"].apply(format_percent_from_decimal)
    merged["Below52_Display"] = merged["Below52_Final"].apply(format_percent_from_decimal)
    merged["PE_Trailing_Display"] = merged["PE_Trailing_Final"].apply(format_pe_value)
    merged["PE_Forward_Display"] = merged["PE_Forward_Final"].apply(format_pe_value)

    return merged.sort_values("Template_Order").reset_index(drop=True)


def build_summary_strip(cards: list[tuple[str, str]]) -> str:
    items = []
    for label, value in cards:
        items.append(
            f'<div class="summary-card"><div class="label">{escape(label)}</div><div class="value">{value}</div></div>'
        )
    return f'<div class="summary-strip">{"".join(items)}</div>'


def zscore_cell_style(z_score: float | None) -> str:
    if z_score is None or pd.isna(z_score):
        return "background: #f7f3ea; color: #6d6457;"
    clipped = float(np.clip(z_score, -2.5, 2.5))
    intensity = abs(clipped) / 2.5
    if clipped >= 0:
        bg = f"rgba(47, 138, 87, {0.12 + 0.30 * intensity:.3f})"
        color = "#165a35"
    else:
        bg = f"rgba(186, 85, 63, {0.12 + 0.30 * intensity:.3f})"
        color = "#8e3628"
    return f"background: {bg}; color: {color};"


def build_factor_state_table(df: pd.DataFrame) -> str:
    rows: list[str] = []
    for _, row in df.iterrows():
        def factor_cell(horizon: str) -> str:
            value = row.get(f"{horizon}_value")
            z_score = row.get(f"{horizon}_z")
            text = "&mdash;" if value is None or pd.isna(value) else f"{value * 100:.1f}%"
            title = "" if z_score is None or pd.isna(z_score) else f" title='Z-Score: {z_score:.2f}'"
            return f"<td class='num factor-cell' style=\"{zscore_cell_style(z_score)}\"{title}>{text}</td>"

        rows.append(
            "<tr>"
            f"<td class='name'>{escape(str(row['Factor']))}</td>"
            f"<td class='num'>{row['Return_Strength']:.1f}</td>"
            f"{factor_cell('21d')}"
            f"{factor_cell('63d')}"
            f"{factor_cell('126d')}"
            f"{factor_cell('252d')}"
            "</tr>"
        )
    return (
        "<section class='group-block'>"
        "<div class='group-header'>State of the Market</div>"
        "<div class='table-scroll'><table class='market-table state-table'>"
        "<thead><tr>"
        "<th>Factor</th><th>Return Strength</th><th>21D</th><th>63D</th><th>126D</th><th>252D</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def build_macro_cards(df: pd.DataFrame) -> str:
    cards: list[str] = []
    for _, row in df.iterrows():
        latest = "&mdash;" if pd.isna(row["Latest"]) else f'{row["Latest"]:.2f}{row["Suffix"]}'
        z_text = "&mdash;" if pd.isna(row["Z_Score"]) else f'{row["Z_Score"]:.2f}'
        as_of = "&mdash;" if pd.isna(row["As_Of"]) else pd.Timestamp(row["As_Of"]).date().isoformat()
        cards.append(
            "<div class='macro-card'>"
            f"<div class='macro-title'>{escape(str(row['Title']))}</div>"
            f"<div class='macro-value'>{latest}</div>"
            f"<div class='macro-regime'>{escape(str(row['Regime']))}</div>"
            f"<div class='macro-meta'>{escape(str(row['Description']))}</div>"
            f"<div class='macro-meta'>10Y Z-Score: {z_text}</div>"
            f"<div class='macro-meta'>As of {as_of}</div>"
            "</div>"
        )
    return f"<div class='macro-grid'>{''.join(cards)}</div>"


def build_curated_table(group_name: str, group_df: pd.DataFrame) -> str:
    rows: list[str] = []
    for _, row in group_df.iterrows():
        rows.append(
            "<tr>"
            f"<td class='name'>{escape(str(row['Name_Final']))}</td>"
            f"<td class='num'>{row['Yield_Display']}</td>"
            f"<td class='num'>{row['YTD_Display']}</td>"
            f"<td class='num'>{row['Return_1Y_Display']}</td>"
            f"<td class='num'>{row['Return_3Y_Display']}</td>"
            f"<td class='num'>{row['Return_5Y_Display']}</td>"
            f"<td class='num'>{row['Below52_Display']}</td>"
            f"<td class='num'>{row['PE_Trailing_Display']}</td>"
            f"<td class='num'>{row['PE_Forward_Display']}</td>"
            f"<td class='master'>{render_master_bar(row['Master_Score'])}</td>"
            "</tr>"
        )
    return (
        f"<section class='group-block'><div class='group-header'>{escape(group_name)}</div>"
        "<div class='table-scroll'><table class='market-table curated-table'>"
        "<thead><tr>"
        "<th>Name</th><th>Yield</th><th>Total Return (YTD)</th><th>Total Return (1Y)</th>"
        "<th>Total Return (3Y)</th><th>Total Return (5Y)</th><th>Below 52W High %</th>"
        "<th>P/E (LTM)</th><th>P/E (NTM)</th><th>Master Score</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def build_universe_table(group_name: str, group_df: pd.DataFrame) -> str:
    rows: list[str] = []
    for _, row in group_df.iterrows():
        rows.append(
            "<tr>"
            f"<td class='signal-col'>{regime_indicator(row['Regime'])}</td>"
            f"<td class='name'>{escape(str(row['Name']))}</td>"
            f"<td class='num'>{row['Yield_Display']}</td>"
            f"<td class='num'>{row['YTD_Display']}</td>"
            f"<td class='num'>{row['Return_1Y_Display']}</td>"
            f"<td class='num'>{row['Return_3Y_Display']}</td>"
            f"<td class='num'>{row['Return_5Y_Display']}</td>"
            f"<td class='num pe'>{row['PE_Display']}</td>"
            f"<td class='master'>{render_master_bar(row['Master_Score'])}</td>"
            "</tr>"
        )
    return (
        f"<section class='group-block'><div class='group-header'>{escape(group_name)}</div>"
        "<div class='table-scroll'><table class='market-table universe-table'>"
        "<thead><tr>"
        "<th>Signal</th><th>Name</th><th>Yield</th><th>YTD</th><th>1Y</th><th>3Y</th><th>5Y</th><th>P/E (LTM/NTM)</th><th>Master Score</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: rgb(0, 0, 0); }
        .block-container { max-width: 1500px; padding-top: 1.5rem; padding-bottom: 2rem; }
        .hero {
            background: rgb(210, 200, 191);
            border: 1px solid rgb(195, 185, 176);
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 10px 30px rgba(150, 140, 131, 0.18);
        }
        .hero h1 { margin: 0; color: rgb(0, 0, 0); font-size: 2rem; font-weight: 700; letter-spacing: 0.01em; }
        .hero p { margin: 0.5rem 0 0 0; color: rgb(150, 140, 131); font-size: 0.98rem; }
        .summary-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 1rem 0 1.25rem 0;
        }
        .summary-card {
            background: rgb(210, 200, 191);
            border: 1px solid rgb(195, 185, 176);
            border-radius: 12px;
            padding: 0.95rem 1rem;
        }
        .summary-card .label {
            color: rgb(150, 140, 131);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .summary-card .value {
            color: rgb(0, 0, 0);
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: 0.2rem;
        }
        .group-block {
            background: rgb(210, 200, 191);
            border: 1px solid rgb(195, 185, 176);
            border-radius: 14px;
            margin-bottom: 1rem;
            overflow: hidden;
            box-shadow: 0 8px 24px rgba(150, 140, 131, 0.16);
        }
        .group-header {
            background: linear-gradient(90deg, rgb(150, 140, 131) 0%, rgb(180, 170, 161) 100%);
            color: rgb(0, 0, 0);
            font-size: 0.95rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 0.8rem 1rem;
            border-bottom: 2px solid rgb(165, 155, 146);
        }
        .table-scroll {
            width: 100%;
            overflow-x: auto;
        }
        .market-table { width: 100%; border-collapse: collapse; }
        .curated-table { min-width: 1360px; }
        .universe-table { min-width: 1080px; }
        .state-table { min-width: 980px; }
        .market-table thead th {
            background: rgb(195, 185, 176);
            color: rgb(0, 0, 0);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 0.8rem 0.85rem;
            border-bottom: 1px solid rgb(180, 170, 161);
            text-align: left;
            white-space: nowrap;
        }
        .market-table tbody td {
            padding: 0.8rem 0.85rem;
            border-bottom: 1px solid rgb(195, 185, 176);
            color: rgb(0, 0, 0);
            font-size: 0.92rem;
            vertical-align: middle;
        }
        .market-table tbody tr:last-child td { border-bottom: none; }
        .market-table tbody tr:hover { background: rgb(195, 185, 176); }
        .name { min-width: 250px; }
        .num, .pe { text-align: right; white-space: nowrap; }
        .signal-col { width: 110px; text-align: center; }
        .master { min-width: 185px; }
        .signal {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 34px;
            height: 34px;
            border-radius: 999px;
            font-size: 1.1rem;
            font-weight: 700;
        }
        .signal.bull { color: #1f7a46; background: rgba(31, 122, 70, 0.12); }
        .signal.bear { color: #b24131; background: rgba(178, 65, 49, 0.12); }
        .signal.neutral { color: #8a7457; background: rgba(138, 116, 87, 0.14); }
        .score-wrap {
            display: grid;
            grid-template-columns: 1fr auto;
            align-items: center;
            gap: 0.6rem;
        }
        .score-bar {
            position: relative;
            height: 10px;
            border-radius: 999px;
            background: rgb(180, 170, 161);
            overflow: hidden;
        }
        .score-bar span {
            display: block;
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, #ba553f 0%, #d2a04c 48%, #2f8a57 100%);
        }
        .score-label { min-width: 38px; text-align: right; font-weight: 700; color: rgb(0, 0, 0); }
        .state-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin-bottom: 1rem;
        }
        .macro-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin-bottom: 1rem;
        }
        .state-card {
            background: rgb(210, 200, 191);
            border: 1px solid rgb(195, 185, 176);
            border-radius: 12px;
            padding: 1rem 1.05rem;
            box-shadow: 0 8px 24px rgba(150, 140, 131, 0.16);
        }
        .macro-card {
            background: rgb(210, 200, 191);
            border: 1px solid rgb(195, 185, 176);
            border-radius: 12px;
            padding: 1rem 1.05rem;
            box-shadow: 0 8px 24px rgba(150, 140, 131, 0.16);
        }
        .state-card .state-label {
            color: rgb(150, 140, 131);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }
        .state-card .state-value {
            color: rgb(0, 0, 0);
            font-size: 1.1rem;
            font-weight: 700;
            line-height: 1.3;
        }
        .macro-title {
            color: rgb(150, 140, 131);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.45rem;
        }
        .macro-value {
            color: rgb(0, 0, 0);
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .macro-regime {
            color: rgb(0, 0, 0);
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .macro-meta {
            color: rgb(150, 140, 131);
            font-size: 0.9rem;
            line-height: 1.4;
        }
        @media (max-width: 1100px) {
            .summary-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .state-grid { grid-template-columns: 1fr; }
            .macro-grid { grid-template-columns: 1fr; }
            .name { min-width: 180px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_curated_dashboard(curated_df: pd.DataFrame, selected_classes: list[str], ticker_query: str) -> None:
    filtered = curated_df[curated_df[CURATED_CLASS_COL].isin(selected_classes)].copy()
    if ticker_query:
        query = ticker_query.lower()
        filtered = filtered[
            filtered["Ticker"].str.lower().str.contains(query, na=False)
            | filtered["Name_Final"].fillna("").str.lower().str.contains(query, na=False)
        ]
    if filtered.empty:
        st.info("No focus assets match the current filters.")
        return
    for group_name in selected_classes:
        group = filtered[filtered[CURATED_CLASS_COL] == group_name]
        if not group.empty:
            st.markdown(build_curated_table(group_name, group), unsafe_allow_html=True)


def render_universe_dashboard(universe_df: pd.DataFrame, selected_classes: list[str], ticker_query: str) -> None:
    filtered = universe_df[universe_df[CLASS_COL].isin(selected_classes)].copy()
    if ticker_query:
        query = ticker_query.lower()
        filtered = filtered[
            filtered["Ticker"].str.lower().str.contains(query, na=False)
            | filtered["Name"].fillna("").str.lower().str.contains(query, na=False)
        ]
    filtered = filtered.sort_values([CLASS_COL, "Master_Score", "Ticker"], ascending=[True, False, True])
    if filtered.empty:
        st.info("No assets match the current filters.")
        return
    for group_name in selected_classes:
        group = filtered[filtered[CLASS_COL] == group_name]
        if not group.empty:
            st.markdown(build_universe_table(group_name, group), unsafe_allow_html=True)


def render_state_market_dashboard(factor_df: pd.DataFrame, factor_query: str) -> None:
    if factor_df.empty:
        st.info("The FactorsToday API could not be reached. The State of the Market dashboard is temporarily unavailable.")
        return

    filtered = factor_df.copy()
    if factor_query:
        query = factor_query.lower()
        filtered = filtered[filtered["Factor"].str.lower().str.contains(query, na=False)]
    if filtered.empty:
        st.info("No factors match the current filters.")
        return
    st.markdown(build_factor_state_table(filtered), unsafe_allow_html=True)


def render_macro_dashboard(macro_df: pd.DataFrame) -> None:
    if macro_df.empty:
        st.info("Add `FRED_API_KEY` to `st.secrets` or your environment to load the macro dashboard.")
        return

    valid = macro_df[macro_df["Latest"].notna()].copy()
    available = len(valid)
    st.markdown(build_macro_cards(macro_df), unsafe_allow_html=True)
    if not available:
        st.warning("FRED data could not be loaded for the configured series. Check your API key and network access.")


def main() -> None:
    inject_css()
    st.markdown(
        """
        <div class="hero">
            <h1>Essential Partners Investment Overview</h1>
            <p>A real-time look at what is happening today's markets.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    raw_df = load_market_data(CSV_FILENAME)
    template_df = load_curated_template()
    scored_df = compute_market_scores(raw_df)
    try:
        factor_historic_df = fetch_factor_historic_snapshot()
        factor_return_panel = fetch_factor_daily_return_panel()
        factor_signal_df = compute_factor_signal_frame(factor_return_panel)
        state_factor_df = (
            factor_historic_df.merge(factor_signal_df, on="Factor", how="inner")
            .loc[lambda df: df["Factor"].isin(BASIC_STATE_FACTORS)]
            .assign(_factor_order=lambda df: df["Factor"].map({name: idx for idx, name in enumerate(BASIC_STATE_FACTORS)}))
            .sort_values("_factor_order")
            .drop(columns="_factor_order")
        )
    except Exception:
        state_factor_df = pd.DataFrame()
    macro_df = fetch_macro_backdrop()
    enrichment_df = fetch_yfinance_enrichment(scored_df["Ticker"].dropna().unique().tolist())
    universe_df = build_universe_frame(scored_df.merge(enrichment_df, on="Ticker", how="left"))
    curated_df = build_curated_frame(scored_df, enrichment_df, template_df)

    curated_classes = template_df[CURATED_CLASS_COL].dropna().drop_duplicates().tolist()
    universe_classes = raw_df[CLASS_COL].dropna().drop_duplicates().tolist()

    with st.sidebar:
        st.header("Dashboard")
        dashboard_mode = st.radio("View", ["Curated Overview", "Full Universe", "State of the Market", "Macro Dashboard"], index=0)
        search_label = "Search Factor" if dashboard_mode == "State of the Market" else "Search Ticker"
        search_placeholder = "Market, Momentum, GoldPrice..." if dashboard_mode == "State of the Market" else "SPY, TLT, GLD..."
        ticker_query = st.text_input(search_label, placeholder=search_placeholder)
        if dashboard_mode == "Curated Overview":
            selected_curated_classes = st.multiselect("Asset Class", options=curated_classes, default=curated_classes)
            selected_universe_classes = universe_classes
        else:
            selected_curated_classes = curated_classes
            if dashboard_mode == "Full Universe":
                selected_universe_classes = st.multiselect("Asset Class", options=universe_classes, default=universe_classes)
            else:
                selected_universe_classes = universe_classes
        st.caption("Live enrichments are cached for 60 minutes via st.cache_data for Streamlit Community Cloud.")

    if yf is None:
        st.warning(
            "yfinance is not installed in this local environment. The dashboard will still render from local files, but live Yield, P/E, and YTD enrichments will stay blank."
        )

    if dashboard_mode == "Curated Overview":
        render_curated_dashboard(curated_df, selected_curated_classes, ticker_query.strip())
    elif dashboard_mode == "Full Universe":
        render_universe_dashboard(universe_df, selected_universe_classes, ticker_query.strip())
    elif dashboard_mode == "State of the Market":
        render_state_market_dashboard(state_factor_df, ticker_query.strip())
    else:
        render_macro_dashboard(macro_df)

    st.caption(
        "Signal arrows and market-state classifications come from the proprietary regime score. Live returns, Yield, and P/E depend on yfinance coverage for each ticker."
    )


if __name__ == "__main__":
    main()
