"""
基于 LLM 的关系抽取器模块。

本模块实现了从非结构化文本中自动提取实体间关系的能力。
关系抽取是知识图谱构建的第二步，需要先通过 EntityExtractor 获取实体列表。

关系抽取的工作流程：
    输入文本 + 实体列表 --> 构造提示词 --> 调用 LLM --> 解析 JSON --> 输出关系列表

例如：
    输入文本："阿司匹林可以治疗头痛，但可能导致胃出血。"
    输入实体：[阿司匹林(Drug), 头痛(Symptom), 胃出血(Symptom)]
    输出关系：
    [
        Relation(source="阿司匹林", target="头痛", type="treats"),
        Relation(source="阿司匹林", target="胃出血", type="causes")
    ]

为什么采用两步抽取（先实体后关系）而不是一步到位？
1. 准确率更高：关系抽取时可以引用具体的实体名称，减少幻觉
2. 灵活性更强：可以独立调整实体抽取和关系抽取的策略
3. 可维护性好：两个模块独立，修改一个不影响另一个
4. 调试方便：可以分别检查实体和关系的抽取质量

关系类型说明（医疗领域）：
- treats（治疗）：药物/手术治疗疾病
  例：[阿司匹林] --(treats)--> [头痛]
- causes（导致）：疾病/因素导致症状或其他疾病
  例：[高血压] --(causes)--> [头痛]
- symptom_of（症状属于）：症状是某种疾病的表现
  例：[头痛] --(symptom_of)--> [感冒]
- used_for（用于）：药物/设备用于某种治疗
  例：[心电监护仪] --(used_for)--> [心脏手术]
- belongs_to（属于）：科室/人员属于某个部门
  例：[心内科] --(belongs_to)--> [心血管中心]
- part_of（是...的一部分）：解剖部位是某个器官的一部分
  例：[左心室] --(part_of)--> [心脏]
- interacts_with（相互作用）：药物之间或药物与食物的相互作用
  例：[华法林] --(interacts_with)--> [阿司匹林]
- contradicts（禁忌/矛盾）：药物禁忌或治疗矛盾
  例：[阿司匹林] --(contradicts)--> [出血性疾病]
"""

import json
import logging

from schemas.graph import RELATION_TYPES, Entity, Relation

logger = logging.getLogger(__name__)

# ============================================================
# 关系抽取提示词模板
# ============================================================
# 这个提示词的设计要点：
# 1. 提供已识别的实体列表：让 LLM 知道要抽取哪些实体之间的关系
# 2. 限定关系类型：只抽取预定义的 8 种关系类型
# 3. 明确方向性：source 是关系的起始，target 是关系的目标
# 4. 要求置信度：让 LLM 评估它对这个关系判断的把握程度
#
# {entities_text}：实体列表的文本表示，每行一个实体
# {relation_types}：关系类型的枚举列表
# {text}：原始文本内容
RELATION_PROMPT = """你是一个专业的医疗领域信息抽取助手。请从以下文本中实体之间的关系。

已识别的实体列表：
{entities_text}

要求：
1. 只提取上述实体之间的关系
2. 关系类型限定为：{relation_types}
3. source 是关系的起始实体，target 是关系的目标实体
4. 返回严格的 JSON 数组格式

文本内容：
{text}

请以 JSON 数组格式返回提取的关系，每个关系包含以下字段：
[
  {{"source": "源实体名称", "target": "目标实体名称", "relation_type": "关系类型", "confidence": 0.9}},
  ...
]

只返回 JSON 数组，不要有任何其他文字："""


class RelationExtractor:
    """
    基于 LLM 的关系抽取器。

    给定文本和已提取的实体列表，使用结构化 JSON 提示词抽取实体间关系。

    使用示例：
        # 创建抽取器
        extractor = RelationExtractor(llm_client, settings)

        # 假设已有实体列表
        entities = [
            Entity(name="阿司匹林", type="Drug", ...),
            Entity(name="头痛", type="Symptom", ...),
        ]

        # 抽取关系
        relations = extractor.extract(
            text="阿司匹林可以治疗头痛。",
            entities=entities,
            doc_id="doc1"
        )

        # 返回结果
        # [
        #     Relation(
        #         source_id="doc1::entity::阿司匹林::Drug",
        #         target_id="doc1::entity::头痛::Symptom",
        #         relation_type="treats",
        #         confidence=0.9
        #     )
        # ]
    """

    def __init__(self, llm_client, settings=None):
        """
        初始化关系抽取器。

        Args:
            llm_client: LLM 客户端实例
            settings: 可选的 Settings 配置对象
        """
        self._llm = llm_client
        self._settings = settings

    def extract(
        self, text: str, entities: list[Entity], doc_id: str, chunk_id: str | None = None
    ) -> list[Relation]:
        """
        从文本中抽取实体间的关系。

        处理流程：
        1. 检查输入（文本为空或实体少于2个时无法抽取关系）
        2. 将实体列表格式化为文本，作为 LLM 的上下文
        3. 构造提示词
        4. 调用 LLM 获取响应
        5. 解析 JSON 响应
        6. 将实体名称映射为实体 ID
        7. 去重并返回结果

        Args:
            text: 原始文本内容
            entities: 已提取的实体列表（来自 EntityExtractor）
            doc_id: 来源文档 ID
            chunk_id: 来源分片 ID（可选）

        Returns:
            抽取到的关系列表。如果抽取失败，返回空列表。
        """
        # 前置检查：至少需要2个实体才能抽取关系
        if not text or not text.strip() or len(entities) < 2:
            return []

        # 将实体列表格式化为文本
        # 格式："- 高血压（Disease）"
        # 这样 LLM 可以清楚地知道有哪些实体可以建立关系
        entities_text = "\n".join(
            f"- {e.name}（{e.type}）" for e in entities
        )

        # 构造提示词
        prompt = RELATION_PROMPT.format(
            entities_text=entities_text,
            relation_types="、".join(RELATION_TYPES),
            text=text[:3000],  # 截断避免超出上下文
        )

        # 构造对话消息
        messages = [
            {"role": "system", "content": "你是一个信息抽取助手，只输出JSON格式的结果。"},
            {"role": "user", "content": prompt},
        ]

        try:
            # 调用 LLM
            response = self._llm.chat(messages, temperature=0.1, max_tokens=2048)

            # 解析响应
            relations = self._parse_response(response, entities, doc_id, chunk_id)
            logger.debug("Extracted %d relations from doc %s", len(relations), doc_id)
            return relations

        except Exception:
            logger.warning("Relation extraction failed for doc %s", doc_id, exc_info=True)
            return []

    def _parse_response(
        self,
        response: str,
        entities: list[Entity],
        doc_id: str,
        chunk_id: str | None,
    ) -> list[Relation]:
        """
        解析 LLM 返回的 JSON 响应。

        关键步骤：将 LLM 输出的实体名称映射为实体 ID。
        LLM 返回的是实体名称（如"阿司匹林"），但关系需要存储实体 ID（如"doc1::entity::阿司匹林::Drug"）。

        映射策略：
        1. 精确匹配：实体名称完全一致
        2. 模糊匹配：名称包含关系（处理 LLM 输出的名称可能不完全一致的情况）

        Args:
            response: LLM 的原始响应文本
            entities: 已提取的实体列表，用于名称到 ID 的映射
            doc_id: 来源文档 ID
            chunk_id: 来源分片 ID

        Returns:
            解析后的关系列表
        """
        text = response.strip()

        # 提取 JSON 数组
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            logger.warning("No JSON array found in LLM response: %s", text[:200])
            return []

        json_str = text[start : end + 1]

        try:
            items = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response: %s", json_str[:200])
            return []

        if not isinstance(items, list):
            return []

        # 构建实体名称到 ID 的映射
        # 例如：{"阿司匹林": "doc1::entity::阿司匹林::Drug", ...}
        entity_map = {e.name: e.id for e in entities}

        relations = []
        for item in items:
            if not isinstance(item, dict):
                continue

            source_name = item.get("source", "").strip()
            target_name = item.get("target", "").strip()
            relation_type = item.get("relation_type", "").strip()

            # 跳过不完整的数据
            if not source_name or not target_name or not relation_type:
                continue

            # 验证关系类型
            if relation_type not in RELATION_TYPES:
                continue

            # 查找实体 ID（先精确匹配，再模糊匹配）
            source_id = entity_map.get(source_name)
            target_id = entity_map.get(target_name)

            if not source_id:
                source_id = self._fuzzy_match(source_name, entity_map)
            if not target_id:
                target_id = self._fuzzy_match(target_name, entity_map)

            # 如果找不到实体，跳过这个关系
            if not source_id or not target_id:
                logger.debug(
                    "Skipping relation: %s -> %s (entity not found)",
                    source_name, target_name
                )
                continue

            # 解析置信度
            confidence = item.get("confidence", 0.8)
            if not isinstance(confidence, (int, float)):
                confidence = 0.8
            confidence = max(0.0, min(1.0, float(confidence)))

            relations.append(
                Relation(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    confidence=confidence,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                )
            )

        # 去重：基于 (source_id, target_id, relation_type)
        seen = set()
        unique_relations = []
        for r in relations:
            key = (r.source_id, r.target_id, r.relation_type)
            if key not in seen:
                seen.add(key)
                unique_relations.append(r)

        return unique_relations

    def _fuzzy_match(self, name: str, entity_map: dict[str, str]) -> str | None:
        """
        模糊匹配实体名称。

        当精确匹配失败时，尝试模糊匹配。
        例如：LLM 输出"阿司匹"，可以匹配到"阿司匹林"。

        Args:
            name: 要匹配的名称
            entity_map: 实体名称到 ID 的映射

        Returns:
            匹配到的实体 ID，如果没有匹配则返回 None
        """
        name_lower = name.lower()
        for entity_name, entity_id in entity_map.items():
            if name_lower in entity_name.lower() or entity_name.lower() in name_lower:
                return entity_id
        return None
