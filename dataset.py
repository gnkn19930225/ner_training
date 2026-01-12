"""
Dataset 模組 - 處理資料集與 tokenization
"""
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class NERDataset(Dataset):
    """NER 資料集"""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[List[str]],
        tokenizer: AutoTokenizer,
        label2id: Dict[str, int],
        max_length: int = 512
    ):
        """
        初始化 NER 資料集
        
        Args:
            texts: 文本列表
            labels: 標籤列表 (字元級別)
            tokenizer: Tokenizer
            label2id: 標籤到 ID 的映射
            max_length: 最大序列長度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        char_labels = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        # 對齊標籤到 token
        offset_mapping = encoding.pop("offset_mapping")[0]
        token_labels = self._align_labels(char_labels, offset_mapping)
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(token_labels, dtype=torch.long)
        }
    
    def _align_labels(
        self,
        char_labels: List[str],
        offset_mapping: torch.Tensor
    ) -> List[int]:
        """
        將字元級別標籤對齊到 token 級別
        
        Args:
            char_labels: 字元級別標籤
            offset_mapping: Token 的字元偏移映射
            
        Returns:
            Token 級別的標籤 ID 列表
        """
        token_labels = []
        
        for start, end in offset_mapping.tolist():
            if start == 0 and end == 0:
                # 特殊 token ([CLS], [SEP], [PAD])
                token_labels.append(-100)
            elif start < len(char_labels):
                # 使用第一個字元的標籤
                label = char_labels[start]
                token_labels.append(self.label2id.get(label, self.label2id["O"]))
            else:
                token_labels.append(-100)
        
        return token_labels


def build_label_mappings(labels: List[List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    建立標籤映射
    
    Args:
        labels: 所有標籤列表
        
    Returns:
        label2id: 標籤到 ID 的映射
        id2label: ID 到標籤的映射
    """
    unique_labels = set()
    for label_seq in labels:
        unique_labels.update(label_seq)
    
    # 確保 O 標籤在第一位
    unique_labels.discard("O")
    sorted_labels = ["O"] + sorted(unique_labels)
    
    label2id = {label: idx for idx, label in enumerate(sorted_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    return label2id, id2label
