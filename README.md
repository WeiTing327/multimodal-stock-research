<div align="center">

# Multimodal Stock Research

### Financial AI Research Platform  
### for Image + Text + Time-Series Learning

K-line Image Classification • Financial Deep Learning • Multimodal AI

</div>

---

![](reports/confusion_matrices/resnet18_image_only_confusion_matrix.png)

---

## Updates

### 2026-05
- Added ResNet18 baseline model
- Added CUDA GPU training support
- Added data augmentation pipeline
- Improved image size from 224 → 256
- Added experiment reproducibility via random seed
- Added confusion matrix evaluation
- Completed first financial image classification baseline

### Next Updates
- Multi-ticker support (2330 / 0050 / 2308)
- Financial backtesting engine
- Ubuntu deployment
- Daily automated update pipeline
- Netlify dashboard
- Multimodal fusion (Image + Text + Macro)

---

# Overview

This project aims to build a complete AI financial research platform combining:

- Financial chart images (Computer Vision)
- Financial news & sentiment analysis (NLP / LLM)
- Macro & time-series data (Time Series Modeling)

The current stage focuses on:

> Financial Image Classification using K-line candlestick charts and CNN-based models.

---

# Project Goal

The goal of this project is not only building prediction models, but creating a:

```text
Continuous Research + Automated Update + Visualized Financial AI Platform
```

The long-term target includes:

- Deep learning research
- Multimodal learning
- Financial prediction
- Backtesting
- Automated deployment
- Daily dashboard system

---

## Baseline Analysis

初步 image-only baseline 結果顯示，
單純使用 K 線圖預測短期股價方向具有明顯挑戰。

模型在訓練集能取得較高準確率，
但在測試集表現不穩定，
反映金融時間序列資料存在：

- 高噪音（High Noise）
- 市場分布轉移（Distribution Shift）
- 非穩態特性（Non-stationary）

因此，後續研究將擴展至：

- 多標的資料
- 技術指標
- 財經新聞文本
- 總經資料

以提升模型泛化能力與實務應用價值。