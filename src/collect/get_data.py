from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


def fetch_2330_price(years: int = 5) -> pd.DataFrame:
    """
    抓取 2330.TW 過去幾年的日股價資料。
    處理 yfinance 可能回傳的 MultiIndex 欄位。
    """
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=365 * years + 30)

    ticker = "2330.TW"
    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError("抓不到 2330.TW 的資料，請檢查網路或 ticker。")

    # 先把 index 轉回欄位
    df = df.reset_index()

    # 如果欄位是 MultiIndex，先壓平成單層
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if col[0] != "" else col[1]
            for col in df.columns
        ]

    # 統一欄位名稱格式
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]

    # 確保有 date 欄位
    if "date" not in df.columns:
        raise KeyError(f"找不到 date 欄位，實際欄位為：{df.columns.tolist()}")

    # 只保留需要的欄位
    keep_cols = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    existing_cols = [col for col in keep_cols if col in df.columns]
    df = df[existing_cols].copy()

    # 日期格式整理
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # 數值欄位轉 numeric
    numeric_cols = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 去重、排序
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    return df


def save_price_csv(df: pd.DataFrame, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")


def main():
    output_path = "data/raw/price_2330.csv"

    df = fetch_2330_price(years=5)
    save_price_csv(df, output_path)

    print("✅ 2330 price data saved successfully!")
    print(f"Output: {output_path}")
    print(f"Rows: {len(df)}")
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nPreview:")
    print(df.head())
    print("\nTail:")
    print(df.tail())


if __name__ == "__main__":
    main()