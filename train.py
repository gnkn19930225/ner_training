"""
主訓練腳本
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from config import ModelConfig, TrainingConfig
from data_loader import MongoDataLoader, parse_ner_data
from dataset import NERDataset, build_label_mappings
from trainer import NERTrainer


def set_seed(seed: int) -> None:
    """設定隨機種子以確保可重現性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 載入設定
    model_config = ModelConfig()

    training_config = TrainingConfig()
    
    # 設定隨機種子
    set_seed(training_config.random_seed)
    
    print("=" * 80)
    print("NER 訓練程式")
    print("=" * 80)
    print(f"模型: {model_config.model_name}")
    print(f"裝置: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()
    
    # 從 MongoDB 載入資料
    print("正在從 MongoDB 載入資料...")
    with MongoDataLoader() as loader:
        documents = loader.fetch_data()
    
    print(f"載入了 {len(documents)} 筆資料")
    
    if len(documents) == 0:
        print("錯誤: 沒有找到符合條件的資料")
        return
    
    # 解析 NER 資料
    print("正在解析 NER 資料...")
    texts, labels = parse_ner_data(documents)
    print(f"解析完成，共 {len(texts)} 筆有效資料")
    
    # 建立標籤映射
    label2id, id2label = build_label_mappings(labels)
    print(f"標籤數量: {len(label2id)}")
    print(f"標籤: {list(label2id.keys())}")
    
    # 更新標籤數量
    num_labels = len(label2id)
    
    # 隨機切分訓練與測試資料
    print(f"\n正在切分資料 (測試集比例: {training_config.test_size})...")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=training_config.test_size,
        random_state=training_config.random_seed
    )
    print(f"訓練集: {len(train_texts)} 筆")
    print(f"測試集: {len(test_texts)} 筆")
    
    # 初始化訓練器
    print("\n正在初始化模型...")
    trainer = NERTrainer(
        model_config=model_config,
        training_config=training_config,
        label2id=label2id,
        id2label=id2label
    )
    
    # 建立資料集
    train_dataset = NERDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=trainer.tokenizer,
        label2id=label2id,
        max_length=model_config.max_length
    )
    
    test_dataset = NERDataset(
        texts=test_texts,
        labels=test_labels,
        tokenizer=trainer.tokenizer,
        label2id=label2id,
        max_length=model_config.max_length
    )
    
    # 建立 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False
    )
    
    # 開始訓練
    print("\n" + "=" * 80)
    print("開始訓練")
    print("=" * 80)
    history = trainer.train(train_loader, test_loader)
    
    # 印出分類報告
    trainer.print_classification_report(test_loader)
    
    # 印出錯誤結果
    trainer.print_errors(test_texts, test_loader)
    
    # 儲存模型
    if training_config.save_best_model:
        trainer.save_model(training_config.output_dir)
    
    print("\n訓練完成!")


if __name__ == "__main__":
    main()
