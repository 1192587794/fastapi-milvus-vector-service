"""
Query 改写器模块。

本模块实现了 RAG 流水线中的 Query 改写功能，在召回之前将用户的原始问题
改写成更适合检索的形式，提高召回率和检索精度。

支持的改写策略：

1. 查询扩展（Query Expansion）
   - 将一个问题扩展成多个语义相关的子问题
   - 对每个子问题分别进行向量召回，合并去重
   - 优势：增加召回率，覆盖更多相关文档
   - 示例：
     原始问题："高血压怎么治疗？"
     扩展结果：
     - "高血压的治疗方法有哪些？"
     - "高血压患者应该吃什么药？"
     - "如何控制高血压？"

2. HyDE（Hypothetical Document Embeddings）
   - 让 LLM 生成一个假设性答案
   - 用假设性答案的 embedding 去检索，而不是用问题的 embedding
   - 优势：答案与文档的语义更接近，提高检索精度
   - 论文：https://arxiv.org/abs/2212.10496
   - 示例：
     原始问题："什么是向量数据库？"
     假设性答案："向量数据库是一种专门用于存储和检索高维向量的数据库系统..."
     用这个答案的 embedding 去检索，比用问题的 embedding 更精准

3. Step-back Prompting
   - 将具体问题改写成更抽象/通用的问题
   - 优势：找到更广泛的背景知识，提供更好的上下文
   - 论文：https://arxiv.org/abs/2310.06117
   - 示例：
     原始问题："高血压患者能吃阿司匹林吗？"
     Step-back："高血压患者的用药禁忌有哪些？"

4. 关键词提取（Keyword Extraction）
   - 从问题中提取关键词
   - 用关键词进行 BM25 检索
   - 优势：补充语义检索的不足，提高精确匹配能力
   - 示例：
     原始问题："阿司匹林的副作用有哪些？"
     关键词：["阿司匹林", "副作用"]

为什么需要 Query 改写？

1. 语义鸿沟：用户的问题和文档的表述方式可能不同
   - 用户问："这个药能治啥病？"
   - 文档写："适应症：用于治疗..."

2. 信息不足：用户问题可能过于简短，缺乏关键信息
   - 用户问："高血压怎么办？"
   - 更好的检索query："高血压的治疗方法、用药方案和生活方式调整"

3. 多意图：复杂问题可能涉及多个子问题
   - 用户问："糖尿病的症状和治疗方法？"
   - 应该分解为两个子问题分别检索

本模块的设计原则：
1. 配置驱动：通过配置开关控制是否启用、使用哪种策略
2. 优雅降级：改写失败时回退到原始问题
3. 零依赖：复用现有的 LLM 客户端，不需要额外安装
4. 可组合：多种策略可以同时使用
"""

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ============================================================
# 提示词模板
# ============================================================

# 查询扩展提示词（带上下文版本）
# 任务：将一个问题扩展成多个语义相关的子问题
# 要求：保持原始意图，但从不同角度提问
# 注意：如果有对话历史，需要结合上下文理解问题的真实意图
QUERY_EXPANSION_PROMPT = """你是一个查询改写助手。请将以下问题扩展成{count}个语义相关但表述不同的子问题。

要求：
1. 保持原始问题的核心意图
2. 从不同角度、用不同方式提问
3. 包含同义词、近义词的替代表达
4. 如果有对话历史，需要结合上下文理解问题的真实意图（如指代消解、意图延续）
5. 返回 JSON 数组格式

{history_context}
原始问题：{question}

请以 JSON 数组格式返回扩展的问题：
["子问题1", "子问题2", "子问题3"]

只返回 JSON 数组，不要有任何其他文字："""


# HyDE 提示词（带上下文版本）
# 任务：生成一个假设性答案
# 要求：答案应该是可能出现在文档中的段落，而不是简短的回答
# 注意：如果有对话历史，需要结合上下文生成更准确的答案
HYDE_PROMPT = """你是一个知识库文档生成助手。请根据以下问题，生成一段可能出现在相关文档中的假设性答案。

要求：
1. 答案应该像真实的文档片段，而不是简短的回答
2. 包含专业术语和详细信息
3. 长度约 100-200 字
4. 使用与问题相同的语言
5. 如果有对话历史，需要结合上下文理解问题的真实意图

{history_context}
问题：{question}

假设性文档片段："""


# Step-back Prompting 提示词（带上下文版本）
# 任务：将具体问题改写成更抽象的问题
# 要求：保留核心概念，但扩大查询范围
# 注意：如果有对话历史，需要结合上下文理解问题的真实意图
STEPBACK_PROMPT = """你是一个查询改写助手。请将以下具体问题改写成一个更抽象、更通用的问题。

要求：
1. 保留问题的核心概念和领域
2. 扩大查询范围，获取更广泛的背景知识
3. 改写后的问题应该更容易在文档中找到答案
4. 如果有对话历史，需要结合上下文理解问题的真实意图（如指代消解）
5. 只返回改写后的问题，不要有其他文字

{history_context}
原始问题：{question}

改写后的抽象问题："""


# 关键词提取提示词（带上下文版本）
# 任务：从问题中提取关键词
# 要求：提取名词、动词、专业术语等有检索价值的词
# 注意：如果有对话历史，需要结合上下文提取更准确的关键词
KEYWORD_PROMPT = """你是一个关键词提取助手。请从以下问题中提取有检索价值的关键词。

要求：
1. 提取名词、动词、专业术语
2. 去除停用词（的、是、在、了、吗、呢等）
3. 保留实体名称（人名、地名、机构名、产品名等）
4. 如果有对话历史，需要结合上下文提取关键词（如上一轮提到的实体）
5. 返回 JSON 数组格式

{history_context}
问题：{question}

请以 JSON 数组格式返回关键词：
["关键词1", "关键词2", "关键词3"]

只返回 JSON 数组，不要有任何其他文字："""


@dataclass
class RewrittenQuery:
    """
    改写后的查询数据结构。

    包含原始问题和各种改写结果，调用方可以根据配置选择使用哪些。

    属性说明：
        original: 原始用户问题
        expanded: 扩展的子问题列表（查询扩展策略）
        hyde_answer: 假设性答案（HyDE 策略）
        stepback: 抽象化问题（Step-back 策略）
        keywords: 关键词列表（关键词提取策略）
    """

    original: str = ""
    expanded: list[str] = field(default_factory=list)
    hyde_answer: str = ""
    stepback: str = ""
    keywords: list[str] = field(default_factory=list)


class QueryRewriter:
    """
    Query 改写器。

    支持四种改写策略，可以通过配置控制使用哪些策略。

    使用示例：
        # 创建改写器
        rewriter = QueryRewriter(llm_client, settings)

        # 改写问题
        rewritten = rewriter.rewrite("高血压怎么治疗？")

        # 使用改写结果
        for sub_query in rewritten.expanded:
            results = retrieve(sub_query)  # 对每个子问题分别召回
        """

    def __init__(self, llm_client, settings=None):
        """
        初始化 Query 改写器。

        Args:
            llm_client: LLM 客户端实例
            settings: 可选的 Settings 配置对象
        """
        self._llm = llm_client
        self._settings = settings

    def rewrite(
        self,
        question: str,
        history: list[dict] | None = None,
        strategy: str = "all",
    ) -> RewrittenQuery:
        """
        改写用户问题。

        根据配置的策略，对问题进行改写。
        如果某个策略失败，会记录警告并跳过，不影响其他策略。

        Args:
            question: 用户原始问题
            history: 对话历史（可选，用于上下文感知的改写）
                     格式：[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            strategy: 改写策略
                     - "expansion": 只做查询扩展
                     - "hyde": 只做 HyDE
                     - "stepback": 只做 Step-back
                     - "keywords": 只做关键词提取
                     - "all": 全部策略

        Returns:
            RewrittenQuery 对象，包含各种改写结果
        """
        result = RewrittenQuery(original=question)

        # 构建上下文信息
        history_context = self._build_history_context(history)

        # 查询扩展
        if strategy in ("expansion", "all"):
            try:
                result.expanded = self._expand_query(question, history_context)
                logger.debug("Query expansion: %s -> %s", question, result.expanded)
            except Exception:
                logger.warning("Query expansion failed", exc_info=True)

        # HyDE
        if strategy in ("hyde", "all"):
            try:
                result.hyde_answer = self._generate_hyde(question, history_context)
                logger.debug("HyDE: %s -> %s", question, result.hyde_answer[:50])
            except Exception:
                logger.warning("HyDE generation failed", exc_info=True)

        # Step-back
        if strategy in ("stepback", "all"):
            try:
                result.stepback = self._stepback(question, history_context)
                logger.debug("Step-back: %s -> %s", question, result.stepback)
            except Exception:
                logger.warning("Step-back failed", exc_info=True)

        # 关键词提取
        if strategy in ("keywords", "all"):
            try:
                result.keywords = self._extract_keywords(question, history_context)
                logger.debug("Keywords: %s -> %s", question, result.keywords)
            except Exception:
                logger.warning("Keyword extraction failed", exc_info=True)

        return result

    def _build_history_context(self, history: list[dict] | None) -> str:
        """
        构建对话历史上下文字符串。

        将对话历史格式化为 LLM 可理解的上下文信息，
        用于帮助 LLM 理解用户问题的真实意图。

        Args:
            history: 对话历史列表

        Returns:
            格式化的上下文字符串，如果没有历史则返回空字符串
        """
        if not history:
            return ""

        # 只取最近 3 轮对话（6 条消息），避免上下文过长
        recent_history = history[-6:] if len(history) > 6 else history

        context_parts = ["对话历史："]
        for msg in recent_history:
            role = "用户" if msg.get("role") == "user" else "助手"
            content = msg.get("content", "")
            # 截断过长的内容
            if len(content) > 200:
                content = content[:200] + "..."
            context_parts.append(f"{role}：{content}")

        return "\n".join(context_parts) + "\n\n"

    def _expand_query(self, question: str, history_context: str = "", count: int = 3) -> list[str]:
        """
        查询扩展：生成多个语义相关的子问题。

        通过 LLM 将一个问题扩展成多个不同表述的子问题，
        每个子问题可能匹配到不同的文档，从而增加召回率。

        Args:
            question: 原始问题
            history_context: 对话历史上下文
            count: 扩展数量

        Returns:
            扩展的子问题列表
        """
        if self._settings:
            count = getattr(self._settings, "query_expansion_count", count)

        prompt = QUERY_EXPANSION_PROMPT.format(
            question=question,
            count=count,
            history_context=history_context,
        )

        messages = [
            {"role": "system", "content": "你是一个查询改写助手，只输出JSON格式的结果。"},
            {"role": "user", "content": prompt},
        ]

        response = self._llm.chat(messages, temperature=0.7, max_tokens=512)
        return self._parse_json_list(response)

    def _generate_hyde(self, question: str, history_context: str = "") -> str:
        """
        HyDE：生成假设性答案。

        让 LLM 生成一个可能出现在相关文档中的假设性答案，
        然后用这个答案的 embedding 去检索。

        为什么用答案而不是问题去检索？
        - 问题和文档的语义可能有差异（语义鸿沟）
        - 答案与文档的语义更接近，因为它们都是"陈述句"
        - 论文证明这种方式可以显著提高检索精度

        Args:
            question: 原始问题
            history_context: 对话历史上下文

        Returns:
            假设性答案文本
        """
        prompt = HYDE_PROMPT.format(
            question=question,
            history_context=history_context,
        )

        messages = [
            {"role": "system", "content": "你是一个知识库文档生成助手。"},
            {"role": "user", "content": prompt},
        ]

        response = self._llm.chat(messages, temperature=0.7, max_tokens=512)
        return response.strip()

    def _stepback(self, question: str, history_context: str = "") -> str:
        """
        Step-back：生成更抽象的问题。

        将具体问题改写成更抽象、更通用的问题，
        用于检索更广泛的背景知识。

        示例：
        - "高血压患者能吃阿司匹林吗？" -> "高血压患者的用药禁忌有哪些？"
        - "Python 的 list 和 tuple 有什么区别？" -> "Python 的数据结构有哪些？"
        - "GPT-4 的参数量是多少？" -> "大语言模型的参数规模和性能关系？"

        Args:
            question: 原始问题
            history_context: 对话历史上下文

        Returns:
            抽象化的问题
        """
        prompt = STEPBACK_PROMPT.format(
            question=question,
            history_context=history_context,
        )

        messages = [
            {"role": "system", "content": "你是一个查询改写助手。"},
            {"role": "user", "content": prompt},
        ]

        response = self._llm.chat(messages, temperature=0.3, max_tokens=256)
        return response.strip()

    def _extract_keywords(self, question: str, history_context: str = "") -> list[str]:
        """
        关键词提取：提取问题中的关键词。

        提取的关键词用于：
        1. BM25 检索：用关键词进行精确匹配
        2. 混合召回：补充语义检索的不足

        Args:
            question: 原始问题
            history_context: 对话历史上下文

        Returns:
            关键词列表
        """
        prompt = KEYWORD_PROMPT.format(
            question=question,
            history_context=history_context,
        )

        messages = [
            {"role": "system", "content": "你是一个关键词提取助手，只输出JSON格式的结果。"},
            {"role": "user", "content": prompt},
        ]

        response = self._llm.chat(messages, temperature=0.1, max_tokens=256)
        return self._parse_json_list(response)

    def _parse_json_list(self, response: str) -> list[str]:
        """
        解析 LLM 返回的 JSON 数组。

        LLM 的输出可能不完全是 JSON，这个方法会尝试提取数组部分。

        Args:
            response: LLM 的原始响应

        Returns:
            解析后的字符串列表
        """
        text = response.strip()

        # 找到 JSON 数组的开始和结束
        start = text.find("[")
        end = text.rfind("]")

        if start == -1 or end == -1:
            logger.warning("No JSON array found in response: %s", text[:200])
            return []

        json_str = text[start : end + 1]

        try:
            items = json.loads(json_str)
            if isinstance(items, list):
                # 过滤非字符串元素
                return [str(item).strip() for item in items if item]
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON: %s", json_str[:200])

        return []
