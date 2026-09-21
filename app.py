import os
import pandas as pd
import yfinance as yf


def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculates MACD and Signal lines."""
    df = df.copy()
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    return df


def main():
    # Load tickers
    tickers_file = "tickers.txt"
    if not os.path.exists(tickers_file):
        print(f"Error: {tickers_file} not found.")
        return

    with open(tickers_file, "r") as f:
        tickers = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]

    if not tickers:
        print("No valid tickers found.")
        return

    print(f"Fetching bulk market data for {len(tickers)} symbols...")

    # Single bulk request to prevent Yahoo HTTP 429 Rate Limits
    try:
        data = yf.download(
            tickers=tickers,
            period="250d",
            interval="1d",
            group_by="ticker",
            threads=True,
            progress=False
        )
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return

    results = []

    for sym in tickers:
        try:
            # Extract ticker DataFrame safely
            if len(tickers) == 1:
                df = data.dropna()
            else:
                if sym not in data.columns.levels[0]:
                    continue
                df = data[sym].dropna()

            if len(df) < 35:
                continue

            df = calculate_macd(df)

            curr_macd = df["MACD"].iloc[-1]
            curr_sig = df["Signal"].iloc[-1]
            prev_macd = df["MACD"].iloc[-2]
            prev_sig = df["Signal"].iloc[-2]
            close = df["Close"].iloc[-1]

            # Signal logic
            if prev_macd <= prev_sig and curr_macd > curr_sig:
                status = "🟢 BULLISH CROSSOVER"
            elif prev_macd >= prev_sig and curr_macd < curr_sig:
                status = "🔴 BEARISH CROSSOVER"
            elif curr_macd > curr_sig:
                status = "🟢 Bullish Trend"
            else:
                status = "🔴 Bearish Trend"

            results.append({
                "Ticker": sym,
                "Close": f"${close:.2f}",
                "MACD": f"{curr_macd:.4f}",
                "Signal": f"{curr_sig:.4f}",
                "Status": status
            })
        except Exception:
            continue

    df_res = pd.DataFrame(results)

    # Format output as Markdown for GitHub Actions UI
    summary_md = "## 📊 Daily MACD Screener Results\n\n"
    if not df_res.empty:
        summary_md += df_res.to_markdown(index=False)
    else:
        summary_md += "No signals were generated."

    # Write directly to GitHub Action Step Summary
    github_summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if github_summary_path:
        with open(github_summary_path, "a", encoding="utf-8") as f:
            f.write(summary_md)
    else:
        print("\n" + summary_md)


if __name__ == "__main__":
    main()
