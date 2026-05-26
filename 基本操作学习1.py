from pymilvus import MilvusClient,DataType

# 链接到本地数据库
client = MilvusClient(uri="http://localhost:19530")


import random

def generate_embeddings(num_vectors, dimension = 768):
    return [[random.random() for _ in range(dimension)] for _ in range(num_vectors)]

# 准备数据
data = [
    {
        "title": "Apple iPhone 15 Pro",
        "embedding": generate_embeddings(1)[0],
        "price": 999.99
    },
    {
        "title": "Samsung Galaxy S24",
        "embedding": generate_embeddings(1)[0],
        "price": 899.99
    },
    {
        "title": "Sony WH-1000XM5",
        "embedding": generate_embeddings(1)[0],
        "price": 349.99
    }
]

insert_result = client.insert(
    collection_name="products",
    data=data
)

print(f"Inserted IDs: {insert_result}")