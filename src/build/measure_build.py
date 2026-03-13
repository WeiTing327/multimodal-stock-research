from pathlib import Path
import pandas as pd


def load_price_data(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    # 基本檢查
    required_cols = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要欄位: {missing_cols}")

    # 日期格式整理
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def make_future_return_and_label(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    建立 future_horizon_day_return 與 binary label
    label:
        1 -> future return > 0
        0 -> future return <= 0
    """
    df = df.copy()

    # 往後看 horizon 天的 close
    df[f"future_close_{horizon}d"] = df["close"].shift(-horizon)

    # future return
    df[f"future_{horizon}d_return"] = (
        df[f"future_close_{horizon}d"] - df["close"]
    ) / df["close"]

    # binary label
    df["label"] = (df[f"future_{horizon}d_return"] > 0).astype(int)

    # 最後幾筆因為沒有 future close，不能當訓練樣本
    df = df.dropna(subset=[f"future_{horizon}d_return"]).reset_index(drop=True)

    return df


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_file, index=False)


def main():
    input_path = "data/raw/price_2330.csv"
    output_path = "data/processed/daily_dataset.parquet"

    df = load_price_data(input_path)
    df = make_future_return_and_label(df, horizon=5)
    save_dataset(df, output_path)

    print("daily dataset 建立成功！")
    print(f"Output: {output_path}")
    print(f"Rows: {len(df)}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nLabel distribution:")
    print(df["label"].value_counts(dropna=False))

    print("\nPreview:")
    print(df.head())

    print("\nTail:")
    print(df.tail())


if __name__ == "__main__":
    main()