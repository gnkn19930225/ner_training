"""
資料載入模組 - 負責從 MongoDB 獲取資料
"""
import os
from typing import List, Dict, Tuple, Any
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv

from config import MongoConfig


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
                enterprise_id=os.getenv("MONGO_ENTERPRISE_ID")
            )
        
        self.config = config
        self.client = None
        self.db = None
        self.collection = None
    
    def connect(self) -> None:
        """建立 MongoDB 連線"""
        self.client = MongoClient(self.config.connection_string)
        self.db = self.client[self.config.database_name]
        self.collection = self.db[self.config.collection_name]
    
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
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


def parse_ner_data(documents: List[Dict]) -> Tuple[List[str], List[List[str]]]:
    """
    解析 NER 資料格式
    
    Args:
        documents: MongoDB 文件列表
        
    Returns:
        texts: 文本列表
        labels: 標籤列表 (BIO 格式)
    """
    texts = []
    labels = []
    
    for doc in documents:
        # 取得原始文本
        text = doc.get("OriginMessage", "")
        
        if not text:
            continue
        
        # 從 AnnotationData.CharacterTags 取得 BIO 標籤
        annotation_data = doc.get("AnnotationData", {})
        character_tags = annotation_data.get("CharacterTags", [])
        
        if not character_tags:
            continue
        
        # 按照 CharPosition 排序並提取 BIOTag
        sorted_tags = sorted(character_tags, key=lambda x: x.get("CharPosition", 0))
        char_labels = [tag.get("BIOTag", "O") for tag in sorted_tags]
        
        # 確保標籤數量與文本長度一致
        if len(char_labels) != len(text):
            # 如果不一致，用 O 補齊或截斷
            if len(char_labels) < len(text):
                char_labels.extend(["O"] * (len(text) - len(char_labels)))
            else:
                char_labels = char_labels[:len(text)]
        
        texts.append(text)
        labels.append(char_labels)
    
    return texts, labels
