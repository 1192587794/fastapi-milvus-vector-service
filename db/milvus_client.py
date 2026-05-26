from pathlib import Path

from pymilvus import MilvusClient

from core.config import Settings


class MilvusManager:
    """
    统一管理 MilvusClient 和集合初始化逻辑。

    这层的目标是把“连接数据库”和“准备集合”的职责从业务层里剥离出去，
    避免 service 层既要关心业务，又要关心数据库生命周期。
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._prepare_local_storage_dir()
        client_kwargs = {
            "uri": settings.resolved_milvus_uri,
            "db_name": settings.milvus_db_name,
        }
        if settings.milvus_token:
            client_kwargs["token"] = settings.milvus_token
        self._client = MilvusClient(**client_kwargs)

    def _prepare_local_storage_dir(self) -> None:
        """
        当使用 Milvus Lite 时，提前创建数据库文件所在目录。

        Milvus Lite 会把 `uri` 当作本地数据库文件路径使用。
        如果父目录不存在，客户端初始化会直接报错，因此这里在启动阶段主动兜底创建目录。
        """
        uri = self.settings.resolved_milvus_uri
        if uri.startswith(("http://", "https://")):
            return

        Path(uri).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    @property
    def client(self) -> MilvusClient:
        """对外暴露统一的客户端实例。"""
        return self._client

    def ensure_collection(self) -> None:
        """
        确保目标集合存在。

        模板项目把集合自动初始化放在启动流程中，目的有两个：
        - 降低首次运行门槛，避免用户先手动创建集合；
        - 让集合定义和代码逻辑保持一致，不容易出现“代码更新了，库里结构没跟上”的情况。
        """
        collection_name = self.settings.milvus_collection

        if self.settings.milvus_drop_existing_on_start and self._client.has_collection(collection_name):
            self._client.drop_collection(collection_name)

        if self._client.has_collection(collection_name):
            return

        # 这里采用“字符串主键 + 向量字段 + 动态字段”的方式，
        # 既能让结构足够简单，又能保留 metadata 扩展空间。
        self._client.create_collection(
            collection_name=collection_name,
            dimension=self.settings.milvus_vector_dimension,
            primary_field_name="doc_id",
            id_type="string",
            vector_field_name="embedding",
            metric_type=self.settings.milvus_metric_type,
            auto_id=False,
            enable_dynamic_field=True,
            consistency_level=self.settings.milvus_consistency_level,
        )

    def describe_collection(self) -> dict:
        """返回集合信息，便于健康检查和调试。"""
        return self._client.describe_collection(self.settings.milvus_collection)
