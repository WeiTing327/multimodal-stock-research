from pathlib import Path

import pandas as pd
import mplfinance as mpf


LOOKBACK = 20
PRICE_PATH = "data/raw/price_2330.csv"
DATASET_PATH = "data/processed/daily_dataset.parquet"
OUTPUT_DIR = "data/processed/kline_images"


def load_data():
    price_df = pd.read_csv(PRICE_PATH)
    dataset_df = pd.read_parquet(DATASET_PATH)

    price_df["date"] = pd.to_datetime(price_df["date"])
    dataset_df["date"] = pd.to_datetime(dataset_df["date"])

    price_df = price_df.sort_values("date").reset_index(drop=True)
    dataset_df = dataset_df.sort_values("date").reset_index(drop=True)

    return price_df, dataset_df


def make_output_dir():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def generate_single_kline_image(window_df: pd.DataFrame, output_path: str):
    """
    用 20 日 OHLC 畫一張簡潔白底 K 線圖
    台股風格：上漲紅、下跌綠
    """
    plot_df = window_df.copy()
    plot_df = plot_df.set_index("date")
    plot_df.index.name = "Date"

    plot_df = plot_df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })

    mc = mpf.make_marketcolors(
        up="red",        # 上漲紅色
        down="green",    # 下跌綠色
        edge="inherit",
        wick="inherit",
        volume="inherit"
    )

    style = mpf.make_mpf_style(
        base_mpf_style="classic",
        marketcolors=mc,
        facecolor="white",
        edgecolor="black",
        figcolor="white",
        gridstyle=""
    )

    mpf.plot(
        plot_df,
        type="candle",
        style=style,
        volume=False,
        figsize=(2.56, 2.56),
        axisoff=True,
        tight_layout=True,
        savefig=dict(
            fname=output_path,
            dpi=100,
            bbox_inches="tight",
            pad_inches=0
        )
    )

def generate_all_images():
    price_df, dataset_df = load_data()
    make_output_dir()

    generated = 0
    skipped = 0

    for _, row in dataset_df.iterrows():
        current_date = row["date"]

        # 找到目前日期在原始 price_df 的位置
        matched_idx = price_df.index[price_df["date"] == current_date]

        if len(matched_idx) == 0:
            skipped += 1
            continue

        idx = matched_idx[0]

        # 必須有足夠的 20 日 window
        if idx < LOOKBACK - 1:
            skipped += 1
            continue

        window_df = price_df.iloc[idx - LOOKBACK + 1: idx + 1].copy()

        # double check
        if len(window_df) != LOOKBACK:
            skipped += 1
            continue

        output_name = f"2330_{current_date.strftime('%Y-%m-%d')}.png"
        output_path = str(Path(OUTPUT_DIR) / output_name)

        try:
            generate_single_kline_image(window_df, output_path)
            generated += 1
        except Exception as e:
            print(f"❌ Failed on {current_date.date()}: {e}")
            skipped += 1

    print("✅ K-line image generation finished!")
    print(f"Generated: {generated}")
    print(f"Skipped: {skipped}")
    print(f"Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_all_images()