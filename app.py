import os
import sys
import pandas as pd
import yfinance as yf


def calculate_indicators(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    sma_period: int = 21,
) -> pd.DataFrame:
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


def load_tickers(file_path: str = "nas100.xlsx") -> list:
    """Loads tickers from Excel or text file safely."""
    if os.path.exists(file_path):
        try:
            tickers_df = pd.read_excel(file_path)
            col_name = (
                tickers_df.columns[0]
                if "Ticker" not in tickers_df.columns and "Symbol" not in tickers_df.columns
                else ("Ticker" if "Ticker" in tickers_df.columns else "Symbol")
            )
            return tickers_df[col_name].dropna().astype(str).str.strip().tolist()
        except Exception as e:
            print(f"Warning: Could not parse '{file_path}': {e}")

    if os.path.exists("tickers.txt"):
        with open("tickers.txt", "r") as f:
            return [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]

    return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]


def write_github_summary(res_df: pd.DataFrame, timeframe: str):
    """Outputs formatted Markdown results directly to the GitHub Action Summary UI/App."""
    github_summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not github_summary_path:
        return

    summary_md = []
    summary_md.append(f"## 📈 MACD Screener Results (`{timeframe}`) Page Summary\n")

    if res_df.empty:
        summary_md.append("⚠️ **No stocks matched the screening criteria for this run.**\n")
    else:
        # Add visual badges to signals
        res_df_display = res_df.copy()
        res_df_display["Signal"] = res_df_display["Signal"].apply(
            lambda x: "🟢 **BULL**" if x == "BULL" else "🔴 **BEAR**"
        )

        summary_md.append(f"Found **{len(res_df)} signal(s)**:\n")
        summary_md.append(res_df_display.to_markdown(index=False))
        summary_md.append("\n")

    with open(github_summary_path, "a", encoding="utf-8") as f:
        f.write("\n".join(summary_md))


def run_macd_screener(
    excel_path: str = "nas100.xlsx",
    timeframe: str = "1d",
    min_dist_pct: float = 0.75,
    max_candle_age: int = 2,
):
    tickers = load_tickers(excel_path)

    tf_period_map = {
        "15m": "30d",
        "30m": "60d",
        "1h": "100d",
        "1d": "250d",
        "1wk": "2y",
    }
    data_period = tf_period_map.get(timeframe, "250d")

    results = []

    print(f"\nScanning {len(tickers)} tickers on '{timeframe}' timeframe (Candle Lookback 0 to {max_candle_age})...")

    for ticker in tickers:
        try:
            df = yf.download(
                ticker, period=data_period, interval=timeframe, progress=False
            )

            if df.empty or len(df) < 30:
                continue

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

                    results.append(
                        {
                            "Ticker": ticker,
                            "Signal": signal_type,
                            "Candles_Ago": age,
                            "Cross_Date": date_stamp,
                            "Close": round(close_price, 2),
                            "MACD_Pct_of_Price": round(curr_macd_pct, 2),
                            "SMA21_Pos": sma21_pos,
                        }
                    )
                    break

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    res_df = pd.DataFrame(results)

    # Output to terminal
    if not res_df.empty:
        print(f"\nScan complete! Found {len(results)} signal(s).")
        print(res_df.to_string(index=False))
    else:
        print("\nScan complete. No stocks matched the criteria.")

    # Output to GitHub Summary UI / App
    write_github_summary(res_df, timeframe)


if __name__ == "__main__":
    selected_timeframe = sys.argv[1] if len(sys.argv) > 1 else "1d"

    run_macd_screener(
        excel_path="nas100.xlsx",
        timeframe=selected_timeframe,
        min_dist_pct=0.75,
        max_candle_age=2,
    )