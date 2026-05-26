from pymilvus import MilvusClient,DataType

# 链接到本地数据库
client = MilvusClient(uri="http://localhost:19530")

# 创建Schema
schema = MilvusClient.create_schema(
    auto_id = True,
    enable_dynamic_field = True,
)

# 添加主键字段
schema.add_field(field_name="id",datatype=DataType.INT64,is_primary=True)

# 添加向量字段
schema.add_field(field_name="title",datatype=DataType.VARCHAR, max_length=255)

# 添加标量字段
schema.add_field(field_name="embedding",datatype=DataType.FLOAT_VECTOR,dim=768)

schema.add_field(field_name="price",datatype=DataType.FLOAT)

# 创建带索引的Collection
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="embedding",
    index_type="IVF_FLAT",
    metric_type="COSINE",
    params={"nlist": 128}
)


client.create_collection(collection_name="products",schema=schema,index_params=index_params)

