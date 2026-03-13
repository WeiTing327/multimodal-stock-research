from pathlib import Path
import pandas as pd


DATASET_PATH = "data/processed/daily_dataset.parquet"
IMAGE_DIR = Path("data/processed/kline_images")
OUTPUT_DIR = Path("data/splits")


def main():
    # 讀 dataset
    df = pd.read_parquet(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 建 image_path
    df["image_filename"] = df["date"].dt.strftime("2330_%Y-%m-%d.png")
    df["image_path"] = df["image_filename"].apply(lambda x: str(IMAGE_DIR / x))

    # 只保留真的有圖的資料
    df["image_exists"] = df["image_path"].apply(lambda x: Path(x).exists())
    df = df[df["image_exists"]].copy().reset_index(drop=True)

    # 保留需要的欄位
    split_df = df[["date", "image_path", "label"]].copy()

    # 時間切分：70 / 15 / 15
    n = len(split_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = split_df.iloc[:train_end].copy()
    val_df = split_df.iloc[train_end:val_end].copy()
    test_df = split_df.iloc[val_end:].copy()

    # 建輸出資料夾
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 存檔
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False, encoding="utf-8-sig")

    # 顯示資訊
    print("✅ Image splits created successfully!")
    print(f"Total samples with images: {n}")
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")

    print("\nTrain date range:")
    print(train_df['date'].min(), "to", train_df['date'].max())

    print("\nVal date range:")
    print(val_df['date'].min(), "to", val_df['date'].max())

    print("\nTest date range:")
    print(test_df['date'].min(), "to", test_df['date'].max())

    print("\nLabel distribution:")
    print("Train:")
    print(train_df["label"].value_counts())
    print("\nVal:")
    print(val_df["label"].value_counts())
    print("\nTest:")
    print(test_df["label"].value_counts())


if __name__ == "__main__":
    main()