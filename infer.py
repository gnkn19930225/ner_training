"""
ONNX NER 推論腳本
用法: python infer.py --model ./output/onnx --text "你的訂單文字"
"""
import re
import json
import argparse
import numpy as np


def normalize_text(text: str) -> str:
    return re.sub(r'[\uFE00-\uFE0F]', '', text)


def run_inference(model_dir: str, text: str, max_length: int = 512):
    import onnxruntime as ort
    from transformers import AutoTokenizer

    # 載入 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    session = ort.InferenceSession(f"{model_dir}/model.onnx")

    # 載入 id2label
    with open(f"{model_dir}/config.json", encoding="utf-8") as f:
        config = json.load(f)
    id2label = {int(k): v for k, v in config["id2label"].items()}

    # 正規化文字
    text = normalize_text(text)
    print(f"文字長度: {len(text)} 字元")

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=False
    )

    actual_len = int(inputs["attention_mask"].sum()) - 2  # 扣掉 CLS/SEP
    print(f"實際 token 數: {actual_len}（max_length={max_length}）")
    if actual_len >= max_length - 2:
        print("⚠️  警告: 文字可能被截斷！請增加 max_length")

    # 推論
    outputs = session.run(None, {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    })
    logits = outputs[0][0]  # shape: (seq_len, num_labels)
    pred_ids = np.argmax(logits, axis=-1)

    # 還原結果（跳過 CLS/SEP/PAD）
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    results = []
    current_entity = None
    current_chars = []

    for i, (token, pred_id) in enumerate(zip(tokens, pred_ids)):
        if token in ("[CLS]", "[SEP]", "[PAD]"):
            if current_entity and current_chars:
                results.append((current_entity, "".join(current_chars)))
                current_entity = None
                current_chars = []
            continue

        label = id2label.get(pred_id, "O")
        char = token.replace("##", "")  # 去掉 BERT subword 前綴

        if label.startswith("B-"):
            if current_entity and current_chars:
                results.append((current_entity, "".join(current_chars)))
            current_entity = label[2:]
            current_chars = [char]
        elif label.startswith("I-") and current_entity:
            current_chars.append(char)
        else:
            if current_entity and current_chars:
                results.append((current_entity, "".join(current_chars)))
                current_entity = None
                current_chars = []

    # 印出結果
    print("\n=== NER 結果 ===")
    if results:
        for entity_type, value in results:
            print(f"  {entity_type}: {value}")
    else:
        print("  （未偵測到任何實體）")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./output/onnx", help="ONNX 模型目錄")
    parser.add_argument("--text", type=str, help="要推論的文字")
    parser.add_argument("--max-length", type=int, default=512, help="最大 token 數（預設 512）")
    parser.add_argument("--file", type=str, help="從檔案讀取輸入文字")
    args = parser.parse_args()

    if args.text:
        run_inference(args.model, args.text, args.max_length)
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read().strip()
        run_inference(args.model, text, args.max_length)
    else:
        print("請用 --text 或 --file 指定輸入")
        print("  python infer.py --model ./output/onnx --text \"文字\"")
        print("  python infer.py --model ./output/onnx --file input.txt")


if __name__ == "__main__":
    main()
