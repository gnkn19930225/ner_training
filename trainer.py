"""
訓練模組 - 負責模型訓練邏輯
"""
import os
import glob
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoModelForTokenClassification, AutoTokenizer
from tqdm import tqdm
import numpy as np
from seqeval.metrics import classification_report, f1_score

from config import ModelConfig, TrainingConfig


class EarlyStopping:
    """Early Stopping 機制"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        """
        Args:
            patience: 容忍多少個 epoch 沒有改善
            min_delta: 最小改善幅度
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        檢查是否應該停止訓練
        
        Args:
            score: 當前分數 (越高越好)
            
        Returns:
            是否應該停止
        """
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.should_stop


class NERTrainer:
    """NER 模型訓練器"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        device: str = None
    ):
        """
        初始化訓練器
        
        Args:
            model_config: 模型設定
            training_config: 訓練設定
            label2id: 標籤到 ID 映射
            id2label: ID 到標籤映射
            device: 運算裝置
        """
        self.model_config = model_config
        self.training_config = training_config
        self.label2id = label2id
        self.id2label = id2label
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型和 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_config.model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id
        ).to(self.device)
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=training_config.patience,
            min_delta=training_config.min_delta
        )
        
        # 最佳模型
        self.best_model_state = None
        self.best_f1 = 0.0
        
        # 權重儲存路徑
        self.weights_dir = os.path.join(training_config.output_dir, "weights")
        os.makedirs(self.weights_dir, exist_ok=True)
        
        # 訓練日期
        self.train_date = datetime.now().strftime("%Y%m%d_%H%M%S")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, List[float]]:
        """
        訓練模型
        
        Args:
            train_loader: 訓練資料載入器
            val_loader: 驗證資料載入器
            
        Returns:
            訓練歷史記錄
        """
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )
        
        # Learning rate scheduler - 自動調降
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.training_config.lr_scheduler_factor,
            patience=self.training_config.lr_scheduler_patience,
            verbose=True
        )
        
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
            "learning_rate": []
        }
        
        for epoch in range(self.training_config.num_epochs):
            # 訓練階段
            train_loss = self._train_epoch(train_loader, optimizer)

            # 驗證階段
            val_loss, val_f1, _, _ = self._evaluate(val_loader)

            # 記錄當前 learning rate
            current_lr = optimizer.param_groups[0]["lr"]

            print(f"Epoch {epoch + 1}/{self.training_config.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            print(f"  Learning Rate: {current_lr:.2e}")

            # 保存最佳模型
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.best_model_state = self.model.state_dict().copy()
                self._save_best_weight(val_f1)
                print(f"  ✓ New best model saved (F1: {val_f1:.4f})")

            # 更新 scheduler（可能會降低 learning rate）
            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_f1)
            new_lr = optimizer.param_groups[0]["lr"]

            # 如果 learning rate 有變化，顯示提示
            if new_lr < old_lr:
                print(f"  ⚠ Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")

            # 更新歷史（記錄更新後的 learning rate）
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)
            history["learning_rate"].append(new_lr)

            # Early stopping 檢查
            if self.early_stopping(val_f1):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

            print()
        
        # 載入最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # 儲存最後一個 epoch 的權重
        self._save_last_weight(history["val_f1"][-1] if history["val_f1"] else 0.0)
        
        return history
    
    def _save_best_weight(self, f1_score: float) -> None:
        """儲存最佳權重，刪除舊的最佳權重"""
        # 刪除舊的 best 權重
        old_best_files = glob.glob(os.path.join(self.weights_dir, f"{self.train_date}_best_*.pt"))
        for f in old_best_files:
            os.remove(f)
        
        # 儲存新的 best 權重
        filename = f"{self.train_date}_best_f1_{f1_score:.4f}.pt"
        filepath = os.path.join(self.weights_dir, filename)
        torch.save(self.model.state_dict(), filepath)
        print(f"  權重已儲存: {filename}")
    
    def _save_last_weight(self, f1_score: float) -> None:
        """儲存最後一個 epoch 的權重"""
        filename = f"{self.train_date}_last_f1_{f1_score:.4f}.pt"
        filepath = os.path.join(self.weights_dir, filename)
        torch.save(self.model.state_dict(), filepath)
        print(f"最後權重已儲存: {filename}")
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """訓練一個 epoch"""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        return total_loss / len(train_loader)

    def _evaluate(
        self,
        data_loader: DataLoader
    ) -> Tuple[float, float, List[List[str]], List[List[str]]]:
        """
        評估模型
        
        Returns:
            loss: 平均損失
            f1: F1 分數
            all_preds: 所有預測結果
            all_labels: 所有真實標籤
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                # 取得預測結果
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                # 轉換為標籤
                for pred, label, mask in zip(
                    predictions.cpu().numpy(),
                    labels.cpu().numpy(),
                    attention_mask.cpu().numpy()
                ):
                    pred_labels = []
                    true_labels = []
                    
                    for p, l, m in zip(pred, label, mask):
                        if m == 1 and l != -100:
                            pred_labels.append(self.id2label[p])
                            true_labels.append(self.id2label[l])
                    
                    if pred_labels:
                        all_preds.append(pred_labels)
                        all_labels.append(true_labels)
        
        avg_loss = total_loss / len(data_loader)
        f1 = f1_score(all_labels, all_preds) if all_labels else 0.0
        
        return avg_loss, f1, all_preds, all_labels
    
    def print_errors(
        self,
        texts: List[str],
        data_loader: DataLoader,
        metadata: List[Dict] = None,
        max_errors: int = 20
    ) -> None:
        """
        印出預測錯誤的結果

        Args:
            texts: 原始文本列表
            data_loader: 資料載入器
            metadata: 元數據列表（包含日期等信息）
            max_errors: 最多顯示幾個錯誤
        """
        self.model.eval()
        errors = []
        text_idx = 0
        
        print("\n" + "=" * 80)
        print("預測錯誤分析")
        print("=" * 80)
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                for pred, label, mask, ids in zip(
                    predictions.cpu().numpy(),
                    labels.cpu().numpy(),
                    attention_mask.cpu().numpy(),
                    input_ids.cpu().numpy()
                ):
                    has_error = False
                    error_details = []
                    
                    tokens = self.tokenizer.convert_ids_to_tokens(ids)
                    
                    for i, (p, l, m, token) in enumerate(zip(pred, label, mask, tokens)):
                        if m == 1 and l != -100 and p != l:
                            has_error = True
                            error_details.append({
                                "token": token,
                                "predicted": self.id2label[p],
                                "actual": self.id2label[l]
                            })
                    
                    if has_error and len(errors) < max_errors:
                        text = texts[text_idx] if text_idx < len(texts) else "N/A"
                        meta = metadata[text_idx] if metadata and text_idx < len(metadata) else {}
                        errors.append({
                            "text": text[:100] + "..." if len(text) > 100 else text,
                            "errors": error_details,
                            "metadata": meta
                        })

                    text_idx += 1
        
        # 按照日期排序錯誤
        errors.sort(key=lambda x: x.get("metadata", {}).get("message_date") or "")

        # 印出錯誤
        for i, error in enumerate(errors, 1):
            print(f"\n錯誤 #{i}")

            # 顯示日期信息
            if error.get("metadata"):
                meta = error["metadata"]
                if meta.get("message_date"):
                    print(f"訊息日期: {meta['message_date']}")
                if meta.get("id"):
                    print(f"ID: {meta['id']}")

            print(f"文本: {error['text']}")
            print("錯誤詳情:")
            for detail in error["errors"][:5]:  # 每個樣本最多顯示 5 個錯誤
                print(f"  Token: '{detail['token']}' | 預測: {detail['predicted']} | 實際: {detail['actual']}")

        print(f"\n總共發現 {len(errors)} 個含有錯誤的樣本")
    
    def save_model(self, path: str) -> None:
        """儲存模型"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"模型已儲存至: {path}")
    
    def print_classification_report(self, data_loader: DataLoader) -> None:
        """印出分類報告"""
        _, _, all_preds, all_labels = self._evaluate(data_loader)
        
        print("\n" + "=" * 80)
        print("分類報告")
        print("=" * 80)
        print(classification_report(all_labels, all_preds))
