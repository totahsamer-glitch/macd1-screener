import os
import pandas as pd
import yfinance as yf


def calculate_macd(df, fast=12, slow=26, signal=9):
  df = df.copy()
  ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
  ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
  df["MACD"] = ema_fast - ema_slow
  df["Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
  return df


def main():
  if not os.path.exists("tickers.txt"):
    print("tickers.txt missing")
    return

  with open("tickers.txt", "r") as f:
    tickers = [line.strip().upper() for line in f if line.strip()]

  if not tickers:
    return

  # Single bulk download to prevent Yahoo HTTP rate limits
  data = yf.download(
      tickers=tickers,
      period="250d",
      interval="1d",
      group_by="ticker",
      threads=True,
      progress=False,
  )

  results = []
  for sym in tickers:
    try:
      df = data.copy() if len(tickers) == 1 else data[sym].dropna()
      if len(df) < 35:
        continue

      df = calculate_macd(df)
      c_macd, c_sig = df["MACD"].iloc[-1], df["Signal"].iloc[-1]
      p_macd, p_sig = df["MACD"].iloc[-2], df["Signal"].iloc[-2]
      close = df["Close"].iloc[-1]

      if p_macd <= p_sig and c_macd > c_sig:
        status = "🟢 BULLISH CROSS"
      elif p_macd >= p_sig and c_macd < c_sig:
        status = "🔴 BEARISH CROSS"
      elif c_macd > c_sig:
        status = "🟢 Bullish"
      else:
        status = "🔴 Bearish"

      results.append({
          "Ticker": sym,
          "Close": f"${close:.2f}",
          "MACD": f"{c_macd:.4f}",
          "Signal": f"{c_sig:.4f}",
          "Status": status,
      })
    except Exception:
      continue

  df_res = pd.DataFrame(results)
  summary_md = "## 📊 Daily MACD Screener Summary\n\n"
  summary_md += (
      df_res.to_markdown(index=False)
      if not df_res.empty
      else "No signals generated."
  )

  github_summary_path = os.getenv("GITHUB_STEP_SUMMARY")
  if github_summary_path:
    with open(github_summary_path, "a") as f:
      f.write(summary_md)


if __name__ == "__main__":
  main()
