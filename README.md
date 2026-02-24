# NER Training

基於 Chinese-RoBERTa 的命名實體識別 (NER) 模型訓練專案。

## 環境需求

- Python 3.8+
- CUDA 11.8 (GPU 訓練)

## 安裝

```bash
pip install -r requirements.txt
```

## 專案結構

```
ner_training/
├── config.py          # 設定檔 (模型、訓練參數)
├── data_loader.py     # MongoDB 資料載入
├── dataset.py         # PyTorch Dataset
├── trainer.py         # 訓練器
├── train.py           # 主訓練腳本
├── export_onnx.py     # ONNX 導出工具
└── output/            # 輸出目錄
    ├── weights/       # 模型權重 (.pt)
    └── onnx/          # ONNX 模型
```

## 訓練

### 設定 MongoDB 連線

在 `.env` 檔案中設定 MongoDB 連線資訊：

```env
MONGO_CONNECTION_STRING=mongodb://localhost:27017
MONGO_DATABASE_NAME=your_database
MONGO_COLLECTION_NAME=your_collection
ENTERPRISE_ID=your_enterprise_id
```

### 執行訓練

```bash
python train.py
```

### 訓練參數

可在 `config.py` 中調整：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| model_name | hfl/chinese-roberta-wwm-ext | 基礎模型 |
| max_length | 512 | 最大序列長度 |
| batch_size | 16 | 批次大小 |
| learning_rate | 2e-5 | 學習率 |
| num_epochs | 50 | 訓練輪數 |
| patience | 10 | Early stopping 耐心值 |
| test_size | 0.2 | 測試集比例 |

## 導出 ONNX

訓練完成後，將模型導出為 ONNX 格式以便部署：

### 從權重檔案導出

```bash
python export_onnx.py --weights ./output/weights/20260120_085955_best_f1_0.9097.pt
```

### 完整參數

```bash
python export_onnx.py \
    --weights ./output/weights/xxx.pt \
    --config-path ./output \
    --output-path ./output/onnx \
    --verify
```

### 導出參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| --weights | - | .pt 權重檔案路徑 |
| --model-path | ./output | 完整模型路徑 (未指定 weights 時使用) |
| --config-path | ./output | 模型設定目錄 (包含 config.json) |
| --output-path | ./output/onnx | ONNX 輸出路徑 |
| --base-model | hfl/chinese-roberta-wwm-ext | 基礎模型名稱 |
| --method | optimum | 導出方法: optimum 或 torch |
| --verify | - | 導出後驗證 ONNX 模型 |

### 輸出檔案

導出後 `./output/onnx/` 目錄會包含：

- `model.onnx` - ONNX 模型檔案
- `config.json` - 模型設定 (包含標籤映射)
- `tokenizer_config.json` - Tokenizer 設定
- `vocab.txt` - 詞彙表

## 使用 ONNX 模型推理

**重要：** 推論時必須對輸入文字做與訓練時相同的正規化處理。

```python
import re
import onnxruntime as ort
from transformers import AutoTokenizer


def normalize_text_for_ner(text: str) -> str:
    """移除 variation selectors，與訓練時保持一致"""
    return re.sub(r'[\uFE00-\uFE0F]', '', text)


# 載入 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained("./output/onnx")
session = ort.InferenceSession("./output/onnx/model.onnx")

# 推理 (注意: 必須先正規化文字)
text = "你的測試文字"
text = normalize_text_for_ner(text)  # 重要!
inputs = tokenizer(text, return_tensors="np", padding="max_length", max_length=128, truncation=True)
outputs = session.run(None, {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"]
})
logits = outputs[0]
```
