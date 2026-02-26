"""
資料載入模組 - 負責從 MongoDB 獲取資料
"""
import os
import re
from typing import List, Dict, Tuple, Any
from datetime import datetime, timezone, timedelta
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv

from config import MongoConfig


def normalize_text_and_labels(text: str, char_labels: List[str]) -> Tuple[str, List[str]]:
    """
    正規化文字並同步調整標籤，確保兩者位置對齊。

    訓練時使用此函數（同時處理 text 和 char_labels）。
    推論時使用 normalize_text_for_ner（僅處理 text）。

    處理: Variation Selectors (U+FE00-U+FE0F)
    例如: ⚠️ = ⚠ (U+26A0) + ️ (U+FE0F)，移除 FE0F 時同步移除其對應標籤，
    避免後續字元位置偏移造成標籤錯位。
    """
    normalized_chars = []
    normalized_labels = []
    for i, char in enumerate(text):
        if '\uFE00' <= char <= '\uFE0F':
            continue  # 跳過 variation selector，同時捨棄該位置的標籤
        normalized_chars.append(char)
        normalized_labels.append(char_labels[i] if i < len(char_labels) else 'O')
    return ''.join(normalized_chars), normalized_labels


def normalize_text_for_ner(text: str) -> str:
    """
    正規化文字，移除會造成標註偏移的 Unicode 字元。

    推論時使用此函數，必須與訓練時的正規化邏輯保持一致。

    處理: Variation Selectors (U+FE00-U+FE0F)
    例如: ⚠️ = ⚠ + FE0F，移除後變成 ⚠
    """
    text = re.sub(r'[\uFE00-\uFE0F]', '', text)
    return text


class MongoDataLoader:
    """MongoDB 資料載入器"""
    
    def __init__(self, config: MongoConfig = None):
        """
        初始化 MongoDB 連線

        Args:
            config: MongoDB 設定，若為 None 則從環境變數讀取
        """
        if config is None:
            load_dotenv()
            config = MongoConfig(
                connection_string=os.getenv("MONGO_CONNECTION_STRING"),
                database_name=os.getenv("MONGO_DATABASE_NAME"),
                collection_name=os.getenv("MONGO_COLLECTION_NAME"),
                enterprise_id=os.getenv("MONGO_ENTERPRISE_ID"),
                message_collection_name=os.getenv("MONGO_MESSAGE_COLLECTION_NAME", "Message")
            )

        self.config = config
        self.client = None
        self.db = None
        self.collection = None
        self.message_collection = None
    
    def connect(self) -> None:
        """建立 MongoDB 連線"""
        self.client = MongoClient(self.config.connection_string)
        self.db = self.client[self.config.database_name]
        self.collection = self.db[self.config.collection_name]
        self.message_collection = self.db[self.config.message_collection_name]
    
    def disconnect(self) -> None:
        """關閉 MongoDB 連線"""
        if self.client:
            self.client.close()
    
    def fetch_data(self) -> List[Dict[str, Any]]:
        """
        從 MongoDB 獲取符合條件的資料

        Returns:
            符合條件的文件列表
        """
        if self.collection is None:
            self.connect()

        query = {
            "EnterpriseId": ObjectId(self.config.enterprise_id),
            "AnnotationSource": self.config.annotation_source
        }

        documents = list(self.collection.find(query))
        return documents

    def get_message_date(self, merged_message_id: str) -> str:
        """
        根據 MergedMessageId 從 Message collection 獲取日期

        Args:
            merged_message_id: Message 的 _id (從 MergedMessageIds 中取得)

        Returns:
            MessageDate 字符串，格式如 "20251119"；若未找到則返回 None
        """
        if self.message_collection is None:
            self.connect()

        try:
            # 查詢 Message collection，匹配 Details._id
            message_doc = self.message_collection.find_one({
                "Details._id": ObjectId(merged_message_id)
            })

            if message_doc:
                return message_doc.get("MessageDate")
            return None
        except Exception as e:
            print(f"查詢 Message 失敗 (ID: {merged_message_id}): {e}")
            return None
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


def parse_ner_data(documents: List[Dict], loader: MongoDataLoader = None) -> Tuple[List[str], List[List[str]], List[Dict[str, Any]]]:
    """
    解析 NER 資料格式

    Args:
        documents: MongoDB 文件列表
        loader: MongoDataLoader 實例，用於查詢 Message collection

    Returns:
        texts: 文本列表
        labels: 標籤列表 (BIO 格式)
        metadata: 元數據列表（包含日期等信息）
    """
    texts = []
    labels = []
    metadata = []

    for doc in documents:
        # 取得原始文本
        text = doc.get("OriginMessage", "")

        if not text:
            continue

        # 從 AnnotationData.CharacterTags 取得 BIO 標籤 (基於原始文字位置)
        annotation_data = doc.get("AnnotationData", {})
        character_tags = annotation_data.get("CharacterTags", [])

        if not character_tags:
            continue

        # 按照 CharPosition 排序並提取 BIOTag (基於原始文字)
        sorted_tags = sorted(character_tags, key=lambda x: x.get("CharPosition", 0))
        char_labels = [tag.get("BIOTag", "O") for tag in sorted_tags]

        # 確保標籤數量與原始文本長度一致
        if len(char_labels) != len(text):
            if len(char_labels) < len(text):
                char_labels.extend(["O"] * (len(text) - len(char_labels)))
            else:
                char_labels = char_labels[:len(text)]

        # 正規化文字並同步調整標籤，確保兩者位置對齊
        # 必須在 char_labels 建立後才做，否則 variation selector 被移除後位置會偏移
        text, char_labels = normalize_text_and_labels(text, char_labels)

        # 提取元數據 - 從 MergedMessageIds 獲取正確的日期
        message_date = None
        merged_message_ids = doc.get("MergedMessageIds", [])

        if merged_message_ids and loader:
            # 取第一個 MergedMessageId 去查詢 Message collection
            first_message_id = str(merged_message_ids[0])
            message_date = loader.get_message_date(first_message_id)

        # 如果沒有找到日期，使用 CreatedAt 作為後備
        if not message_date:
            created_at = doc.get("CreatedAt")
            if not created_at and doc.get("_id"):
                created_at = doc.get("_id").generation_time
            message_date = created_at.strftime("%Y%m%d") if created_at else None

        meta = {
            "id": str(doc.get("_id", "")),
            "message_date": message_date,
        }

        texts.append(text)
        labels.append(char_labels)
        metadata.append(meta)

    return texts, labels, metadata
