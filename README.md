1. 建立原始股價資料
執行src/collect/get_data.py

2. 轉換成可訓練資料
src/build/build_daily_dataset.py

3. 產生K線圖
src/build/generate_kline_images.py

4. 建立影像與標籤對應表，並切分資料集
rc/build/make_image_splits.py

5. 訓練模型
src/train/train_image_resnet18.py

src/
├─ collect/
│  └─ get_data.py
│
├─ build/
│  ├─ build_daily_dataset.py
│  ├─ generate_kline_images.py
│  └─ make_image_splits.py
│
├─ train/
│  └─ train_image_resnet18.py
│
├─ eval/
│  ├─ evaluate.py
│  └─ plot_confusion_matrix.py
│
└─ utils/

一、專案簡介
本專案旨在研究金融圖像與多模態資料是否能用於預測股票未來價格方向

目前第一階段聚焦於：
使用 K 線圖影像作為輸入，透過深度學習模型預測未來股價方向
Price Data → Kline Image → CNN → Prediction

標的為：
台積電（2330.TW）

後續將逐步加入：
1.新聞文本情緒分析
2.總體經濟指標
3.多模態融合模型


二、資料建構流程
資料由台積電歷史股價資料建構而成

資料來源：
Yahoo Finance
股票代碼：2330.TW

資料包含：
Open、High、Low、Close、Volume、Adj Close

資料處理流程：
歷史股價資料
      ↓
建立 daily dataset
      ↓
計算未來報酬率
      ↓
建立分類標籤
      ↓
20日 sliding window
      ↓
生成 K 線圖影像
      ↓
建立影像分類資料集


三、Label 定義
目前使用二分類預測問題：
future_return > 0  → label = 1 (上漲)
future_return ≤ 0  → label = 0 (下跌)

目前實驗包含兩種預測 horizon：
Rule 1：未來 5 日報酬


四、K線圖影像資料集
每一筆樣本包含：
過去 20 個交易日

轉成圖像：
K線圖影像

影像規格：
解析度：224 × 224
背景：白色
Ｋ棒：紅漲綠跌


生成流程：
20日 OHLC 價格資料
        ↓
K線圖生成
        ↓
224x224 RGB 影像

目前資料量：
約 1200 張 K 線圖


五、資料切分方式
為避免金融時間序列資料洩漏，本專案時間序列切分方式：
Train：70%
Validation：15%
Test：15%

時間範圍：
Train：2021 ~ 2024
Validation：2024 ~ 2025
Test：2025 ~ 2026


六、Baseline 模型
第一版模型使用：
ResNet18（ImageNet pretrained）

模型架構：
K線圖影像
      ↓
ResNet18 CNN backbone
      ↓
Fully Connected Layer
      ↓
UP / DOWN prediction


七、後續實驗方向
接下來將進行以下實驗：

1.測試不同影像大小對模型訓練的準確度
2.測試不同預測時間長度，長線交易、短線交易
3.不同模型架構比較
4.Text模態加入，進行多模態組合
處理方式：Text Embedding、Sentiment Analysis
模型：BERT / FinBERT、Sentence Transformer
