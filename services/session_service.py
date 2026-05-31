"""
会话管理服务 — 基于 Redis 存储多轮对话历史。

工作原理：
- 每个多轮对话分配一个唯一的 session_id（UUID）
- 对话历史以 JSON 列表形式存储在 Redis 中，key 为 session:{session_id}
- 每次读写都会刷新 TTL（过期时间），长时间不活跃的会话自动清理
- 客户端只需传 session_id，服务端自动管理历史的存取

Redis 数据结构：
    key:   session:a1b2c3d4-...
    value: [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]
    TTL:   3600 秒（可配置）
"""

import json
import logging
import uuid

import redis

logger = logging.getLogger(__name__)


class SessionService:
    """
    会话管理服务，负责对话历史的存取和生命周期管理。

    使用 Redis 作为存储后端，支持：
    - 创建新会话（生成 UUID）
    - 读取历史消息
    - 追加新消息（自动刷新 TTL）
    - 删除会话
    """

    def __init__(self, redis_client: redis.Redis, ttl_seconds: int = 3600) -> None:
        """
        初始化会话服务。

        参数:
            redis_client: redis.Redis 实例
            ttl_seconds: 会话过期时间（秒），默认 3600（1 小时）
        """
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds

    def _session_key(self, session_id: str) -> str:
        """生成 Redis key，格式为 session:{session_id}。"""
        return f"session:{session_id}"

    def create_session(self) -> str:
        """
        创建新会话，返回 session_id。

        此时 Redis 中还没有数据，等到第一条消息追加时才写入。
        """
        session_id = str(uuid.uuid4())
        logger.info("创建新会话: %s", session_id)
        return session_id

    def get_history(self, session_id: str) -> list[dict]:
        """
        从 Redis 加载指定会话的对话历史。

        如果 session_id 不存在或已过期，返回空列表（视为新对话）。

        参数:
            session_id: 会话 ID

        返回:
            消息列表，格式为 [{"role": "user", "content": "..."}, ...]
        """
        key = self._session_key(session_id)
        try:
            data = self.redis.get(key)
            if data is None:
                return []
            # Redis 中存储的是 JSON 字符串，反序列化为 Python 列表
            messages = json.loads(data)
            # 刷新 TTL：读取也算活跃，延长会话生命周期
            self.redis.expire(key, self.ttl_seconds)
            return messages
        except (json.JSONDecodeError, redis.RedisError) as e:
            logger.warning("加载会话历史失败 session_id=%s: %s", session_id, e)
            return []

    def append_message(self, session_id: str, role: str, content: str) -> None:
        """
        向指定会话追加一条消息，并刷新 TTL。

        追加操作是原子的：先读取现有历史，追加新消息，再写回 Redis。
        如果会话不存在，会自动创建。

        参数:
            session_id: 会话 ID
            role: 消息角色，"user" 或 "assistant"
            content: 消息内容
        """
        key = self._session_key(session_id)
        try:
            # 读取现有历史
            data = self.redis.get(key)
            messages: list[dict] = json.loads(data) if data else []

            # 追加新消息
            messages.append({"role": role, "content": content})

            # 写回 Redis 并设置 TTL
            self.redis.setex(key, self.ttl_seconds, json.dumps(messages, ensure_ascii=False))
            logger.debug("追加消息到会话 %s: role=%s, 长度=%d", session_id, role, len(content))
        except redis.RedisError as e:
            logger.error("追加消息失败 session_id=%s: %s", session_id, e)
            raise

    def delete_session(self, session_id: str) -> bool:
        """
        删除指定会话。

        参数:
            session_id: 会话 ID

        返回:
            True 表示成功删除，False 表示会话不存在
        """
        key = self._session_key(session_id)
        try:
            result = self.redis.delete(key)
            if result:
                logger.info("删除会话: %s", session_id)
            return bool(result)
        except redis.RedisError as e:
            logger.error("删除会话失败 session_id=%s: %s", session_id, e)
            return False
