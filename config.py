"""
設定檔 - 存放所有設定參數
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class MongoConfig:
    """MongoDB 連線設定"""
    connection_string: str
    database_name: str
    collection_name: str
    enterprise_id: str
    annotation_source: str = "Manual"
    message_collection_name: str = "Message"


@dataclass
class ModelConfig:
    """模型設定"""
    model_name: str = "hfl/chinese-roberta-wwm-ext"
    max_length: int = 512


@dataclass
class TrainingConfig:
    """訓練設定"""
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 50
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    test_size: float = 0.2
    random_seed: int = 42
    
    # Early stopping 設定
    patience: int = 10  # 10代沒下降就結束訓練
    min_delta: float = 0.001

    # Learning rate scheduler 設定
    lr_scheduler_factor: float = 0.5  # Learning rate 減半
    lr_scheduler_patience: int = 5  # 5代沒下降就降低 learning rate
    
    # 輸出設定
    output_dir: str = "./output"
    save_best_model: bool = True
