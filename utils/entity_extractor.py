"""
基于 LLM 的实体抽取器模块。

本模块实现了从非结构化文本中自动提取医疗领域实体的能力。
核心思想：利用大语言模型（LLM）的理解能力，通过精心设计的提示词（Prompt），
让 LLM 从文本中识别并分类出各种医疗实体。

实体抽取的工作流程：
    输入文本 --> 构造提示词 --> 调用 LLM --> 解析 JSON --> 输出实体列表

例如：
    输入："患者有高血压病史，长期服用阿司匹林100mg，近日出现头痛症状。"
    输出：
    [
        Entity(name="高血压", type="Disease"),
        Entity(name="阿司匹林", type="Drug"),
        Entity(name="头痛", type="Symptom")
    ]

为什么选择 LLM 而不是传统 NER 模型？
1. 零样本能力：不需要标注训练数据，直接使用
2. 灵活性：通过修改 Prompt 可以调整抽取的实体类型和粒度
3. 中文支持：大模型对中文的理解能力远超传统 NER 模型
4. 零依赖：复用现有的 LLM 客户端，不需要额外安装 spaCy/HanLP
5. 领域适应：通过 Prompt 中的示例可以快速适应医疗领域

LLM 抽取的局限性：
1. 速度较慢：每次抽取需要调用 LLM API（通常 1-3 秒）
2. 成本较高：每次抽取消耗 token（如果使用付费 API）
3. 不确定性：LLM 的输出可能有随机性，需要后处理
4. 长文本限制：需要截断过长的文本

本模块的优化策略：
1. 低温度生成：temperature=0.1，减少随机性
2. 结构化输出：要求 LLM 返回 JSON 格式，便于解析
3. 实体去重：基于 (name, type) 去重，避免重复实体
4. 类型校验：验证实体类型是否在预定义列表中
5. 异常容错：LLM 调用失败时返回空列表，不影响主流程
"""

import json
import logging

from schemas.graph import ENTITY_TYPES, Entity

logger = logging.getLogger(__name__)

# ============================================================
# 实体抽取提示词模板
# ============================================================
# 这是实体抽取的核心——精心设计的提示词。
# 提示词的质量直接决定了抽取的效果。
#
# 提示词的设计要点：
# 1. 明确角色：告诉 LLM 它是一个"专业的医疗领域信息抽取助手"
# 2. 明确任务：从文本中提取所有实体
# 3. 约束输出格式：要求返回 JSON 数组
# 4. 限定实体类型：只提取预定义的 8 种类型
# 5. 提供输出示例：让 LLM 知道期望的输出格式
#
# {entity_types} 和 {text} 是占位符，运行时会被替换
EXTRACTION_PROMPT = """你是一个专业的医疗领域信息抽取助手。请从以下文本中提取所有实体。

要求：
1. 只提取有明确语义的实体，不要提取过于泛化的词
2. 实体类型限定为：{entity_types}
3. 返回严格的 JSON 数组格式，不要包含任何其他文本

文本内容：
{text}

请以 JSON 数组格式返回提取的实体，每个实体包含以下字段：
[
  {{"name": "实体名称", "type": "实体类型", "attributes": {{"描述": "简要描述"}}}},
  ...
]

只返回 JSON 数组，不要有任何其他文字："""


class EntityExtractor:
    """
    基于 LLM 的实体抽取器。

    使用结构化 JSON 提示词从文本中提取实体。
    支持 Ollama 和 OpenAI 两种 LLM 后端（通过鸭子类型）。

    使用示例：
        # 创建抽取器
        extractor = EntityExtractor(llm_client, settings)

        # 从文本中抽取实体
        entities = extractor.extract(
            text="患者有高血压，服用阿司匹林治疗。",
            doc_id="doc1"
        )

        # 返回结果
        # [
        #     Entity(id="doc1::entity::高血压::Disease", name="高血压", type="Disease", ...),
        #     Entity(id="doc1::entity::阿司匹林::Drug", name="阿司匹林", type="Drug", ...)
        # ]
    """

    def __init__(self, llm_client, settings=None):
        """
        初始化实体抽取器。

        Args:
            llm_client: LLM 客户端实例
                       需要实现 chat(messages, temperature, max_tokens) 方法
                       可以是 OllamaChatClient 或 OpenAIChatClient
            settings: 可选的 Settings 配置对象
                     未来可能用于配置抽取参数（如 batch_size）
        """
        self._llm = llm_client
        self._settings = settings

    def extract(self, text: str, doc_id: str, chunk_id: str | None = None) -> list[Entity]:
        """
        从文本中抽取实体。

        这是抽取器的主入口方法。处理流程：
        1. 检查输入是否为空
        2. 构造提示词（替换占位符）
        3. 调用 LLM 获取响应
        4. 解析 JSON 响应
        5. 去重并返回结果

        Args:
            text: 待抽取的文本内容
                  如果文本过长（>3000字符），会被截断
            doc_id: 来源文档 ID，用于生成实体 ID
            chunk_id: 来源分片 ID（可选），用于定位实体位置

        Returns:
            抽取到的实体列表。如果抽取失败，返回空列表。
        """
        # 空文本检查
        if not text or not text.strip():
            return []

        # 构造提示词
        # 截断文本到3000字符，避免超出 LLM 的上下文窗口
        prompt = EXTRACTION_PROMPT.format(
            entity_types="、".join(ENTITY_TYPES),
            text=text[:3000],
        )

        # 构造对话消息
        messages = [
            {"role": "system", "content": "你是一个信息抽取助手，只输出JSON格式的结果。"},
            {"role": "user", "content": prompt},
        ]

        try:
            # 调用 LLM
            # temperature=0.1：低温度减少随机性，提高输出稳定性
            # max_tokens=2048：足够生成较长的实体列表
            response = self._llm.chat(messages, temperature=0.1, max_tokens=2048)

            # 解析响应
            entities = self._parse_response(response, doc_id, chunk_id)
            logger.debug("Extracted %d entities from doc %s", len(entities), doc_id)
            return entities

        except Exception:
            # 抽取失败时记录警告并返回空列表
            # 这种设计保证了主流程不会因为抽取失败而中断
            logger.warning("Entity extraction failed for doc %s", doc_id, exc_info=True)
            return []

    def _parse_response(
        self, response: str, doc_id: str, chunk_id: str | None
    ) -> list[Entity]:
        """
        解析 LLM 返回的 JSON 响应。

        LLM 的输出可能不完全是 JSON，可能包含额外的文本。
        这个方法会尝试从响应中提取 JSON 数组部分。

        解析策略：
        1. 找到第一个 '[' 和最后一个 ']'
        2. 提取这两个位置之间的文本作为 JSON
        3. 解析 JSON 并转换为 Entity 对象

        Args:
            response: LLM 的原始响应文本
            doc_id: 来源文档 ID
            chunk_id: 来源分片 ID

        Returns:
            解析后的实体列表
        """
        text = response.strip()

        # 尝试找到 JSON 数组的开始和结束位置
        start = text.find("[")
        end = text.rfind("]")

        if start == -1 or end == -1:
            logger.warning("No JSON array found in LLM response: %s", text[:200])
            return []

        # 提取 JSON 部分
        json_str = text[start : end + 1]

        try:
            items = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response: %s", json_str[:200])
            return []

        # 确保是列表类型
        if not isinstance(items, list):
            return []

        # 转换为 Entity 对象
        entities = []
        for item in items:
            if not isinstance(item, dict):
                continue

            name = item.get("name", "").strip()
            entity_type = item.get("type", "Other").strip()

            # 跳过空名称
            if not name:
                continue

            # 验证实体类型，无效类型归为 "Other"
            if entity_type not in ENTITY_TYPES:
                entity_type = "Other"

            # 生成实体 ID
            # 格式：{doc_id}::entity::{name}::{type}
            entity_id = f"{doc_id}::entity::{name}::{entity_type}"

            entities.append(
                Entity(
                    id=entity_id,
                    name=name,
                    type=entity_type,
                    attributes=item.get("attributes", {}),
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                )
            )

        # 去重：基于 (name, type) 组合
        # 同一个实体可能在文本中出现多次，LLM 可能会重复抽取
        seen = set()
        unique_entities = []
        for e in entities:
            key = (e.name, e.type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(e)

        return unique_entities
