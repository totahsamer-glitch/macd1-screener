import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Configure Web Page Layout
st.set_page_config(
    page_title="MACD Stock Screener",
    page_icon="📈",
    layout="wide",
)

# --- CONFIGURATION CONSTANTS ---
TICKER_FILE = "tickers.txt"
EXCEL_FILE = "nas100.xlsx"

# Mapping timeframes to yfinance parameters
TIMEFRAME_CONFIG = {
    "1 Day": {"interval": "1d", "period": "250d", "unit": "Days"},
    "1 Hour": {"interval": "1h", "period": "100d", "unit": "Hours"},
    "30 Mins": {"interval": "30m", "period": "60d", "unit": "30m Bars"},
    "15 Mins": {"interval": "15m", "period": "30d", "unit": "15m Bars"},
    "1 Week": {"interval": "1wk", "period": "2y", "unit": "Weeks"},
}


def load_tickers():
    """Attempts to read tickers from nas100.xlsx or fallback to tickers.txt."""
    if os.path.exists(EXCEL_FILE):
        try:
            tickers_df = pd.read_excel(EXCEL_FILE)
            col_name = (
                tickers_df.columns[0]
                if "Ticker" not in tickers_df.columns and "Symbol" not in tickers_df.columns
                else ("Ticker" if "Ticker" in tickers_df.columns else "Symbol")
            )
            tickers = tickers_df[col_name].dropna().astype(str).str.strip().tolist()
            return list(dict.fromkeys(tickers))
        except Exception:
            pass

    if os.path.exists(TICKER_FILE):
        with open(TICKER_FILE, "r") as f:
            tickers = [
                line.strip().upper()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
        return list(dict.fromkeys(tickers))

    return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]


def calculate_indicators(df, fast=12, slow=26, signal=9, sma_period=21):
    """Calculates MACD, Signal Line, MACD % distance from price, and 21 SMA."""
    df = df.copy()

    # MACD Calculation
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()

    # MACD percentage distance relative to stock price
    df["MACD_Pct"] = (df["MACD"] / df["Close"]) * 100

    # Simple Moving Average 21
    df["SMA21"] = df["Close"].rolling(window=sma_period).mean()

    return df


@st.cache_data(ttl=900, show_spinner=False)
def run_macd_screener(tickers, timeframe_label, min_dist_pct, max_candle_age):
    if not tickers:
        return pd.DataFrame()

    tf_info = TIMEFRAME_CONFIG[timeframe_label]
    interval = tf_info["interval"]
    period = tf_info["period"]

    results = []

    for ticker in tickers:
        try:
            df = yf.download(
                ticker, period=period, interval=interval, progress=False
            )

            if df.empty or len(df) < 30:
                continue

            # Flatten multi-index columns if returned by yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = calculate_indicators(df)
            total_bars = len(df)

            for age in range(max_candle_age + 1):
                curr_idx = total_bars - 1 - age
                prev_idx = curr_idx - 1

                if curr_idx < 0 or prev_idx < 0:
                    continue

                curr_macd = df["MACD"].iloc[curr_idx]
                prev_macd = df["MACD"].iloc[prev_idx]
                curr_sig = df["Signal"].iloc[curr_idx]
                prev_sig = df["Signal"].iloc[prev_idx]
                curr_macd_pct = df["MACD_Pct"].iloc[curr_idx]
                close_price = df["Close"].iloc[curr_idx]
                sma21_val = df["SMA21"].iloc[curr_idx]
                date_stamp = df.index[curr_idx].strftime("%Y-%m-%d %H:%M")

                bull_cross = (curr_macd > curr_sig) and (prev_macd <= prev_sig)
                bear_cross = (curr_macd < curr_sig) and (prev_macd >= prev_sig)

                signal_type = None

                # BULLISH CONDITION
                if (
                    bull_cross
                    and curr_macd < 0
                    and curr_sig < 0
                    and curr_macd_pct <= -abs(min_dist_pct)
                ):
                    signal_type = "BULL"

                # BEARISH CONDITION
                elif (
                    bear_cross
                    and curr_macd > 0
                    and curr_sig > 0
                    and curr_macd_pct >= abs(min_dist_pct)
                ):
                    signal_type = "BEAR"

                if signal_type:
                    sma21_pos = "Above" if close_price > sma21_val else "Below"
                    decimals = 4 if close_price < 1.0 else 2

                    results.append({
                        "Ticker": ticker,
                        "Signal": signal_type,
                        "Candles Ago": age,
                        "Cross Date": date_stamp,
                        "Close Price": round(close_price, decimals),
                        "MACD % of Price": round(curr_macd_pct, 2),
                        "SMA21": round(sma21_val, decimals) if not pd.isna(sma21_val) else None,
                        "SMA21 Pos": sma21_pos,
                    })
                    break

        except Exception:
            continue

    return pd.DataFrame(results)


# --- STYLING FUNCTIONS ---
def style_signal(val):
    if val == "BULL":
        return "background-color: #1b382b; color: #4eff9e; font-weight: bold;"
    elif val == "BEAR":
        return "background-color: #3d1c1d; color: #ff6b6b; font-weight: bold;"
    return ""


def style_sma_pos(val):
    if val == "Above":
        return "color: #4eff9e; font-weight: bold;"
    elif val == "Below":
        return "color: #ff6b6b; font-weight: bold;"
    return ""


def apply_table_styles(df):
    return df.style.map(style_signal, subset=["Signal"]).map(
        style_sma_pos, subset=["SMA21 Pos"]
    )


# ==================== STREAMLIT UI ====================

st.title("📈 MACD Crossover Screener")
st.caption(
    "Automated detection for MACD zero-bound crossovers with minimum distance filtering."
)

tickers = load_tickers()

# Sidebar Controls
with st.sidebar:
    st.header("Screener Parameters")

    selected_tf = st.selectbox(
        "⏱ Select Timeframe",
        options=list(TIMEFRAME_CONFIG.keys()),
        index=0,  # Default to 1 Day
    )

    min_dist = st.slider(
        "Min MACD % Distance from Zero",
        min_value=0.1,
        max_value=3.0,
        value=0.75,
        step=0.05,
        help="MACD line must be at least this percentage of the stock price away from the zero line.",
    )

    max_age = st.slider(
        "Max Candle Lookback",
        min_value=0,
        max_value=5,
        value=2,
        step=1,
        help="Look for signals occurring within 0 to N candles ago.",
    )

    st.write(f"📁 Tickers Loaded: **{len(tickers)}**")

    st.markdown("---")
    if st.button("🔄 Force Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with st.spinner(f"Scanning {len(tickers)} symbols on {selected_tf} timeframe..."):
    df_results = run_macd_screener(tickers, selected_tf, min_dist, max_age)

time_unit = TIMEFRAME_CONFIG[selected_tf]["unit"]

if not df_results.empty:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Signals Found", len(df_results))
    col2.metric("Bullish Signals", len(df_results[df_results["Signal"] == "BULL"]))
    col3.metric("Bearish Signals", len(df_results[df_results["Signal"] == "BEAR"]))

    st.markdown("---")

    # Format table output
    is_penny = (
        (df_results["Close Price"] < 1.0).any()
        if "Close Price" in df_results.columns
        else False
    )
    price_format = "$%.4f" if is_penny else "$%.2f"

    column_formatting = {
        "Ticker": st.column_config.TextColumn("Ticker"),
        "Signal": st.column_config.TextColumn("Signal Type"),
        "Candles Ago": st.column_config.NumberColumn(f"Candles Ago ({time_unit})"),
        "Cross Date": st.column_config.TextColumn("Cross Date"),
        "Close Price": st.column_config.NumberColumn("Close Price", format=price_format),
        "MACD % of Price": st.column_config.NumberColumn("MACD % Dist", format="%.2f%%"),
        "SMA21": st.column_config.NumberColumn("SMA 21", format=price_format),
        "SMA21 Pos": st.column_config.TextColumn("Price vs SMA 21"),
    }

    styled_df = apply_table_styles(df_results)
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        column_config=column_formatting,
    )

    # Download CSV button
    csv_data = df_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Results CSV",
        data=csv_data,
        file_name="macd_screener_results.csv",
        mime="text/csv",
    )
else:
    st.info(
        f"No MACD crossover signals matched the criteria on the {selected_tf} timeframe."
    )