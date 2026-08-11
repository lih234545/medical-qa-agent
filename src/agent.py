"""
医疗问诊 Agent 核心模块。

包含两部分：
  1. 检索工具 search_medical_knowledge：从 FAISS 向量库检索，按相似度阈值过滤，
     命中则返回带来源的参考资料，未命中则返回统一的“无结果”标记。
  2. 基于通义千问的工具调用型 Agent：遵循严格的“仅基于检索结果回答”规则。

Agent 规则（来自项目需求）：
  - 相似度低于阈值(0.5)的检索结果被视为不相关并过滤；
  - 只能依据检索到的参考知识回答，禁止添加检索结果之外的内容；
  - 当检索工具返回“无相关信息”时，必须原样输出固定拒答话术，
    绝不允许使用大模型自身的预训练知识作答。
"""
import re
import warnings

# LangChain/LangGraph 在导入时会触发一条无害的待废弃提示（allowed_objects），
# 而 langchain_core 自身又会把该类告警强制设为“显示”，导致普通的 ignore 过滤器失效。
# 这里在告警的最终输出环节（showwarning）拦截，仅丢弃这一条特定提示，其余告警照常显示。
_orig_showwarning = warnings.showwarning


def _quiet_showwarning(message, category, filename, lineno, file=None, line=None):
    if "allowed_objects" in str(message):
        return
    _orig_showwarning(message, category, filename, lineno, file, line)


warnings.showwarning = _quiet_showwarning

from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool

from . import config
from .knowledge_base import load_all_documents, load_vectorstore

# 全局向量库单例：首次使用时加载，避免重复读取索引
_vectorstore = None

# 原始文档缓存：用于做标题/关键词匹配兜底，解决短查询只靠向量检索不稳定的问题
_documents = None


def _get_vectorstore():
    """惰性加载 FAISS 向量库（单例）。"""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_vectorstore()
    return _vectorstore


def _get_documents():
    """惰性加载全部原始文档，供关键词兜底检索使用。"""
    global _documents
    if _documents is None:
        _documents = load_all_documents()
    return _documents


def _normalize_text(text: str) -> str:
    """归一化文本，便于进行稳定的中英文关键词匹配。"""
    text = text.lower().strip()
    return re.sub(r"\s+", "", text)


def _keyword_search(query: str):
    """基于标题与正文做关键词兜底检索，优先解决短疾病名/药名查询。"""
    query_norm = _normalize_text(query)
    if len(query_norm) < 2:
        return []

    matches = []
    for doc in _get_documents():
        title = doc.metadata.get("title", "")
        title_norm = _normalize_text(title)
        content_norm = _normalize_text(doc.page_content)

        if query_norm == title_norm:
            matches.append((300, doc, "标题精确匹配"))
        elif title_norm and (title_norm in query_norm or query_norm in title_norm):
            matches.append((200, doc, "标题关键词匹配"))
        elif query_norm in content_norm:
            matches.append((100, doc, "正文关键词匹配"))

    matches.sort(key=lambda item: item[0], reverse=True)
    return [(doc, reason) for _, doc, reason in matches[: config.TOP_K]]


@tool
def search_medical_knowledge(query: str) -> str:
    """检索本地医疗知识库（疾病百科、药品说明书、医患问答），返回与问题相关的参考资料。

    回答任何医疗、健康、疾病、药品相关的问题时，都必须先调用本工具获取参考资料。

    Args:
        query: 用户的医疗相关问题或关键信息。

    Returns:
        命中时返回带来源标注的参考资料文本；未命中时返回无结果标记。
    """
    relevant = []
    seen = set()

    # 先做标题/关键词匹配兜底，提升“痛风”“冠心病”这类短查询的命中率
    for doc, reason in _keyword_search(query):
        doc_key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("title", ""),
        )
        if doc_key not in seen:
            relevant.append((doc, "keyword", reason))
            seen.add(doc_key)

    vectorstore = _get_vectorstore()
    # 返回 (Document, L2距离)。因向量已归一化，距离为平方欧氏距离，
    # 余弦相似度 = 1 - 距离 / 2。
    results = vectorstore.similarity_search_with_score(query, k=config.TOP_K)

    for doc, distance in results:
        similarity = 1 - distance / 2
        doc_key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("title", ""),
        )
        if similarity >= config.SIMILARITY_THRESHOLD and doc_key not in seen:
            relevant.append((doc, "vector", similarity))
            seen.add(doc_key)

    # 无任何结果通过阈值 -> 返回统一标记，交由 Agent 触发拒答
    if not relevant:
        return config.NO_RESULT_FLAG

    # 拼接检索到的参考资料，附带来源与匹配方式，便于溯源与调试
    blocks = []
    for i, (doc, match_type, match_value) in enumerate(relevant, start=1):
        source = doc.metadata.get("source", "未知来源")
        title = doc.metadata.get("title", "")
        match_desc = (
            f"匹配方式：{match_value}"
            if match_type == "keyword"
            else f"相似度：{match_value:.2f}"
        )
        blocks.append(
            f"【参考资料{i}｜来源：{source}｜{title}｜{match_desc}】\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(blocks)


# Agent 的系统提示词：把严格规则固化进去
SYSTEM_PROMPT = f"""你是一名严谨、专业的医疗问诊助手。你必须严格遵守以下规则：

1. 【必须检索】回答任何医疗、健康、疾病、症状、用药相关的问题前，
   都必须先调用 `search_medical_knowledge` 工具检索本地知识库。

2. 【仅基于检索结果回答】你只能依据工具返回的参考资料来组织答案，
   严禁添加参考资料中没有的任何信息，严禁使用你自己的预训练知识进行补充、扩写或推测。
   如果参考资料只提到某个要点而未展开解释，就不要自行解释其原理。

3. 【无结果必须拒答】当工具返回的内容为 “{config.NO_RESULT_FLAG}” 时，
   你绝对禁止使用自己的知识回答，必须原样、完整地回复以下这句话，不得增删任何字：
   “{config.REFUSAL_MESSAGE}”

4. 【回答风格】命中知识库时，用通俗、条理清晰的纯文本中文回答，可用简单的分点，
   但不要使用 emoji 表情符号；必要时说明信息来源，
   并在涉及用药、诊断等关键内容时提醒用户以专业医师意见为准。

请始终以患者安全为最高优先级。"""


def build_agent(verbose: bool = False):
    """构建并返回可执行的医疗问诊 Agent（LangChain 1.x create_agent）。"""
    config.check_api_key()

    llm = ChatTongyi(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
    )

    # create_agent 基于 LangGraph 构建工具调用型 Agent，
    # system_prompt 固化严格规则，tools 提供知识库检索能力。
    return create_agent(
        llm,
        tools=[search_medical_knowledge],
        system_prompt=SYSTEM_PROMPT,
        debug=verbose,
    )


# 复用同一个 Agent 实例，避免每次提问都重建
_agent_executor = None


def ask(question: str, verbose: bool = False) -> str:
    """向 Agent 提问并返回最终回答文本。"""
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = build_agent(verbose=verbose)
    # create_agent 生成的图以 messages 作为输入/输出
    result = _agent_executor.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    # 返回最后一条消息（AI 的最终回答）
    return result["messages"][-1].content
