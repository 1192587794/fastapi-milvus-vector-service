import json
from typing import Optional
from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    connections,
    utility
)

class MilvusSchemaDesigner:
    def __init__(self, uri: str = "http://127.0.0.1:19530", token: str ="root:Milvus"):
        """
        初始化 Milvus连接

        Args：
            uri (str): Milvus服务器的URI地址
            token： 认证token
        """
        self.uri = uri
        self.token = token
        self.client = None
        self.collection_name = "hybrid_search_collection"

    
    def connect(self):
        """
        链接到Milvus服务器
        """
        try:
            print(f"正在链接Milvus服务器：{self.uri}")
            self.client = MilvusClient(uri=self.uri, token=self.token)
            print("成功链接到Milvus服务器")
        except Exception as e:
            print(f"链接Milvus服务器失败：{e}")
            raise e
        
    def check_collection_exists(self) -> bool:
        """
        检查集合是否存在
        """

        try:
            if utility.has_collection(self.collection_name):
                print(f"集合 '{self.collection_name}' 已存在")
                return True
            else:
                print(f"集合 '{self.collection_name}' 不存在")
                return False
        except Exception as e:
            print(f"检查集合存在性时出错：{e}")
            raise e
        
    def drop_existing_collection(self):
        """
        删除已存在的集合
        """
        try:
            if self.check_collection_exists():
                print(f"正在删除集合 '{self.collection_name}'")
                self.client.drop_collection(collection_name=self.collection_name)
                print(f"集合 '{self.collection_name}' 已删除")
        except Exception as e:
            print(f"删除集合时出错：{e}")
            raise e
    
    def create_hybrid_search_schema(self,vector_dimension: int = 1024):
        """
        创建支持混合搜索的Schema

        Args：
            vector_dimension： 向量维度
        """

        print(f"向量维度设置为：{vector_dimension}")

        # 创建Schema
        schema = self.client.create_schema(
            auto_id = False,
            enable_dynamic_field = True
        )

        # 添加基础字段

        schema.add_field(field_name = "id",datatype = DataType.INT64,max_length = 100,is_primary = True)
        schema.add_field(field_name = "embedding",datatype = DataType.FLOAT_VECTOR,dim = vector_dimension)

        schema.add_field(
            field_name = "title",
            datatype = DataType.VARCHAR,
            max_length = 65535,
            enable_analyzer = True,
            analyzer_params = {"tokenizer":"jieba"}
        )

        schema.add_field(
            field_name = "content",
            datatype = DataType.VARCHAR,
            max_length = 65535,
            enable_analyzer = True,
            analyzer_params = {"tokenizer":"jieba"}
        )

        # 添加稀疏向量字段（用于存储 BM25结果）
        schema.add_field(field_name = "title_sparse",dataType = DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="content_sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
        