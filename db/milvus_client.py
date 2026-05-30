from pathlib import Path

from pymilvus import MilvusClient,CollectionSchema, FieldSchema, DataType

from core.config import Settings


class MilvusManager:
    """
    统一管理 MilvusClient 和集合初始化逻辑.

    这层的目标是把"连接数据库"和"准备集合"的职责从业务层里剥离出去,
    避免 service 层既要关心业务,又要关心数据库生命周期.
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
        当使用 Milvus Lite 时,提前创建数据库文件所在目录.

        Milvus Lite 会把 `uri` 当作本地数据库文件路径使用.
        如果父目录不存在,客户端初始化会直接报错,因此这里在启动阶段主动兜底创建目录.
        """
        uri = self.settings.resolved_milvus_uri
        if uri.startswith(("http://", "https://")):
            return

        Path(uri).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    @property
    def client(self) -> MilvusClient:
        """对外暴露统一的客户端实例."""
        return self._client

    def ensure_collection(self) -> None:
        """
        确保目标集合存在并已加载到内存.

        模板项目把集合自动初始化放在启动流程中,目的有两个:
        - 降低首次运行门槛,避免用户先手动创建集合;
        - 让集合定义和代码逻辑保持一致,不容易出现"代码更新了,库里结构没跟上"的情况.
        """
        collection_name = self.settings.milvus_collection

        # 如果配置了启动时重建集合,先删掉旧的
        if self.settings.milvus_drop_existing_on_start and self._client.has_collection(collection_name):
            self._client.drop_collection(collection_name)

        # 集合不存在时才创建,已存在则跳过
        if not self._client.has_collection(collection_name):
            # 这里采用"字符串主键 + 向量字段 + 动态字段"的方式
            fields = [
                # 主键字段:字符串类型,需要指定 max_length
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    max_length=255,
                    is_primary=True,
                    auto_id=False,
                ),
                # 向量字段
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.settings.milvus_vector_dimension,
                ),
                # 其他字段可以通过动态字段自动处理,不需要显式定义
            ]

            # 创建 schema
            schema = CollectionSchema(
                fields=fields,
                enable_dynamic_field=True,  # 启用动态字段,保留 metadata 扩展能力
                description="Document collection with embedding",
            )

            # 创建 collection
            self._client.create_collection(
                collection_name=collection_name,
                schema=schema,
                consistency_level=self.settings.milvus_consistency_level,
            )

            # 创建索引(关键步骤!)
            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                metric_type=self.settings.milvus_metric_type,
                index_type="AUTOINDEX",  # 让 Milvus 自动选择最佳索引
            )
            self._client.create_index(
                collection_name=collection_name,
                index_params=index_params,
            )

        # 无论集合是新建还是已存在,都必须加载到内存才能执行向量搜索.
        # 这行必须在 if 块外面,否则集合已存在时会跳过加载,导致 search 报 "collection not loaded".
        self._client.load_collection(collection_name=collection_name)

    def describe_collection(self) -> dict:
        """返回集合信息,便于健康检查和调试."""
        return self._client.describe_collection(self.settings.milvus_collection)
