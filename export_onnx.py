"""
將訓練好的 NER 模型導出為 ONNX 格式
"""
import os
import json
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


def load_model_from_weights(
    weights_path: str,
    config_path: str = None,
    base_model: str = "hfl/chinese-roberta-wwm-ext"
):
    """
    從 .pt 權重檔案載入模型

    Args:
        weights_path: .pt 權重檔案路徑
        config_path: 模型設定目錄（包含 config.json），若無則使用 base_model
        base_model: 基礎模型名稱
    """
    # 嘗試從 config.json 取得 label 映射
    label2id = None
    id2label = None

    if config_path and os.path.exists(os.path.join(config_path, "config.json")):
        with open(os.path.join(config_path, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
            label2id = config.get("label2id")
            id2label = config.get("id2label")
            if id2label:
                id2label = {int(k): v for k, v in id2label.items()}

    if label2id and id2label:
        print(f"從 config.json 載入標籤映射，共 {len(label2id)} 個標籤")
        model = AutoModelForTokenClassification.from_pretrained(
            base_model,
            num_labels=len(label2id),
            label2id=label2id,
            id2label=id2label
        )
    else:
        print("警告: 找不到標籤映射，請確保 --config-path 指向包含 config.json 的目錄")
        print("使用預設模型架構...")
        model = AutoModelForTokenClassification.from_pretrained(base_model)

    # 載入權重
    print(f"正在載入權重: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)

    # 載入 tokenizer
    if config_path and os.path.exists(os.path.join(config_path, "tokenizer_config.json")):
        tokenizer = AutoTokenizer.from_pretrained(config_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    return model, tokenizer


def export_with_optimum(model_path: str, output_path: str) -> None:
    """使用 optimum 導出 ONNX（推薦方法）"""
    from optimum.onnxruntime import ORTModelForTokenClassification

    print(f"正在從 {model_path} 載入模型...")

    # 載入模型並轉換為 ONNX
    model = ORTModelForTokenClassification.from_pretrained(
        model_path,
        export=True
    )

    # 儲存 ONNX 模型
    model.save_pretrained(output_path)
    print(f"ONNX 模型已儲存至: {output_path}")
    print(f"模型檔案: {output_path}/model.onnx")


def export_from_weights(
    weights_path: str,
    output_path: str,
    config_path: str = None,
    base_model: str = "hfl/chinese-roberta-wwm-ext"
) -> None:
    """從 .pt 權重檔案導出 ONNX"""
    model, tokenizer = load_model_from_weights(weights_path, config_path, base_model)
    model.eval()

    # 建立虛擬輸入
    dummy_input = tokenizer(
        "這是一個測試句子",
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True
    )

    # 確保輸出目錄存在
    os.makedirs(output_path, exist_ok=True)
    onnx_path = os.path.join(output_path, "model.onnx")

    # 導出 ONNX
    print("正在導出 ONNX...")
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=14,
        do_constant_folding=True
    )

    # 儲存 tokenizer 和 config
    tokenizer.save_pretrained(output_path)
    model.config.save_pretrained(output_path)

    print(f"ONNX 模型已儲存至: {onnx_path}")


def export_with_torch(model_path: str, output_path: str) -> None:
    """使用 torch.onnx.export 直接導出（備用方法）"""

    print(f"正在從 {model_path} 載入模型...")

    # 載入模型和 tokenizer
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.eval()

    # 建立虛擬輸入
    dummy_input = tokenizer(
        "這是一個測試句子",
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True
    )

    # 確保輸出目錄存在
    os.makedirs(output_path, exist_ok=True)
    onnx_path = os.path.join(output_path, "model.onnx")

    # 導出 ONNX
    print("正在導出 ONNX...")
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=14,
        do_constant_folding=True
    )

    # 複製 tokenizer 相關檔案
    tokenizer.save_pretrained(output_path)

    # 複製模型設定
    model.config.save_pretrained(output_path)

    print(f"ONNX 模型已儲存至: {onnx_path}")


def verify_onnx(output_path: str) -> None:
    """驗證導出的 ONNX 模型"""
    import onnx

    onnx_path = os.path.join(output_path, "model.onnx")

    print("\n正在驗證 ONNX 模型...")
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("ONNX 模型驗證通過!")

    # 顯示模型資訊
    print(f"\n模型輸入:")
    for input in model.graph.input:
        print(f"  - {input.name}: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}")

    print(f"\n模型輸出:")
    for output in model.graph.output:
        print(f"  - {output.name}")


def main():
    parser = argparse.ArgumentParser(description="導出 NER 模型為 ONNX 格式")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./output",
        help="訓練好的模型路徑 (預設: ./output)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="指定 .pt 權重檔案路徑 (例如: ./output/weights/xxx.pt)"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="./output",
        help="模型設定目錄，包含 config.json 和 tokenizer (預設: ./output)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="hfl/chinese-roberta-wwm-ext",
        help="基礎模型名稱 (預設: hfl/chinese-roberta-wwm-ext)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./output/onnx",
        help="ONNX 輸出路徑 (預設: ./output/onnx)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["optimum", "torch"],
        default="optimum",
        help="導出方法: optimum (推薦) 或 torch (預設: optimum)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="導出後驗證 ONNX 模型"
    )

    args = parser.parse_args()

    # 如果指定了權重檔案，使用權重導出
    if args.weights:
        if not os.path.exists(args.weights):
            print(f"錯誤: 權重檔案不存在: {args.weights}")
            return
        export_from_weights(
            weights_path=args.weights,
            output_path=args.output_path,
            config_path=args.config_path,
            base_model=args.base_model
        )
    else:
        # 檢查模型路徑
        if not os.path.exists(args.model_path):
            print(f"錯誤: 模型路徑不存在: {args.model_path}")
            return

        # 導出
        if args.method == "optimum":
            export_with_optimum(args.model_path, args.output_path)
        else:
            export_with_torch(args.model_path, args.output_path)

    # 驗證
    if args.verify:
        verify_onnx(args.output_path)

    print("\n導出完成!")


if __name__ == "__main__":
    main()
