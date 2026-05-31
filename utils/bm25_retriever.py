"""
BM25 稀疏召回模块

BM25（Best Matching 25）是一种经典的关键词匹配算法，擅长精确匹配查询中的关键词。
与向量召回（语义匹配）互补：向量召回能理解同义词和语义相似，BM25 能精确匹配专有名词和关键词。

本模块用于混合检索场景：同时执行向量召回和 BM25 召回，再用 RRF 融合两路结果。
"""

import logging
import re

import jieba
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# 停用词集合：高频无意义词汇，过滤掉可以提高 BM25 的匹配质量
_STOP_WORDS: set[str] = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
    "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
    "它", "们", "那", "些", "什么", "怎么", "如何", "可以", "能",
    "但是", "但", "而", "如果", "因为", "所以", "这个", "那个",
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall",
    "of", "in", "to", "for", "with", "on", "at", "from", "by",
    "and", "or", "not", "no", "it", "this", "that", "as", "if",
}

# 匹配纯标点和空白的正则
_PUNCT_RE = re.compile(r"^[\s\W]+$", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """
    使用 jieba 分词器对文本进行分词。

    流程：
    1. 用 jieba.cut 做中文分词（搜索引擎模式，粒度更细）
    2. 过滤掉停用词、纯标点、纯空白的 token
    3. 转为小写（英文统一大小写）

    示例：
        "什么是向量数据库？" -> ["什么", "向量", "数据库"]
        "Hello World" -> ["hello", "world"]
    """
    # jieba.cut 的 cut_all=False 为精确模式，适合搜索引擎
    raw_tokens = jieba.cut(text, cut_all=False)
    tokens: list[str] = []
    for token in raw_tokens:
        token = token.strip().lower()
        # 跳过空串、纯标点、停用词
        if not token or _PUNCT_RE.match(token) or token in _STOP_WORDS:
            continue
        tokens.append(token)
    return tokens


class BM25Retriever:
    """
    BM25 稀疏召回器。

    工作流程：
    1. 从 Milvus 中查询所有候选文档（带可选的标量过滤）
    2. 对文档文本做分词，构建 BM25 索引
    3. 对用户查询做分词，用 BM25 算法计算每个文档的相关性分数
    4. 按分数降序返回 top_k 条结果

    注意：每次 retrieve 都会重新从 Milvus 拉取文档并构建索引。
    数据量大时可考虑加缓存（如定时刷新 + 内存缓存）。
    """

    def __init__(self, milvus_client, collection_name: str) -> None:
        """
        初始化 BM25 召回器。

        参数:
            milvus_client: pymilvus.MilvusClient 实例
            collection_name: Milvus 集合名称
        """
        self.milvus_client = milvus_client
        self.collection_name = collection_name

    def retrieve(
        self,
        query: str,
        top_k: int,
        filter_expr: str = "",
    ) -> list[dict]:
        """
        BM25 稀疏召回。

        参数:
            query: 用户查询文本
            top_k: 返回结果数量
            filter_expr: Milvus 标量过滤表达式，如 'source == "upload"'

        返回:
            结果列表，每条包含 id、text、source、metadata、bm25_score
        """
        # 第一步：从 Milvus 拉取所有候选文档
        candidates = self._fetch_candidates(filter_expr)
        if not candidates:
            return []

        # 第二步：对文档文本分词，构建 BM25 索引
        corpus = [_tokenize(doc.get("text", "")) for doc in candidates]
        bm25 = BM25Okapi(corpus)

        # 第三步：对查询分词并计算 BM25 分数
        query_tokens = _tokenize(query)
        scores = bm25.get_scores(query_tokens)

        # 第四步：按分数降序排序，取 top_k
        scored_docs = list(zip(candidates, scores, strict=True))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results: list[dict] = []
        for doc, score in scored_docs[:top_k]:
            results.append({
                "id": doc.get("id", ""),
                "text": doc.get("text", ""),
                "source": doc.get("source"),
                "metadata": doc.get("metadata", {}),
                "bm25_score": float(score),
            })

        return results

    def _fetch_candidates(self, filter_expr: str = "") -> list[dict]:
        """
        从 Milvus 拉取所有候选文档。

        使用 query（非 search）接口，不做向量匹配，只做标量过滤。
        limit 设为 10000，覆盖中小规模数据集。大规模场景需要分页拉取。

        拉取失败时返回空列表并记录警告，不影响主流程。
        """
        try:
            results = self.milvus_client.query(
                collection_name=self.collection_name,
                filter=filter_expr if filter_expr else "",
                output_fields=["id", "text", "source", "metadata"],
                limit=10000,
            )
            return results if results else []
        except Exception:
            logger.warning("BM25 候选文档拉取失败，跳过稀疏召回", exc_info=True)
            return []
