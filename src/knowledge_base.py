"""
本地知识库处理模块。

负责三件事：
  1. 把三种不同格式的原始数据（JSON / Markdown / TXT）解析为统一的 LangChain Document；
  2. 使用通义千问 text-embedding 向量化并构建 FAISS 向量索引；
  3. 从磁盘加载已构建好的 FAISS 索引。

设计要点（PPT 可讲）：
  - 三类数据分别对应“结构化 / 半结构化 / 非结构化”，采用不同解析策略；
  - 每个 Document 都带 metadata（来源、类别、标题），便于回答时溯源；
  - 向量库采用余弦相似度，方便用统一阈值（0.5）过滤不相关内容。
"""
import json
import os
import re
from typing import List

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from . import config


# ---------------------------------------------------------------------------
# 1. 结构化数据：疾病百科 JSON
# ---------------------------------------------------------------------------
def load_disease_documents() -> List[Document]:
    """把疾病百科 JSON 中的每条疾病记录转成一个 Document。"""
    with open(config.DISEASE_JSON, "r", encoding="utf-8") as f:
        diseases = json.load(f)

    documents = []
    for item in diseases:
        # 将结构化字段拼成自然语言文本，便于向量检索与大模型阅读
        content = (
            f"疾病名称：{item['disease_name']}\n"
            f"所属科室：{item['department']}\n"
            f"定义：{item['definition']}\n"
            f"典型症状：{'、'.join(item['typical_symptoms'])}\n"
            f"常见病因：{'、'.join(item['causes'])}\n"
            f"治疗原则：{'、'.join(item['treatment_principles'])}\n"
            f"饮食禁忌：{'、'.join(item['dietary_taboos'])}"
        )
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": "疾病百科",
                    "category": "disease",
                    "title": item["disease_name"],
                },
            )
        )
    return documents


# ---------------------------------------------------------------------------
# 2. 半结构化数据：药品说明书 Markdown
# ---------------------------------------------------------------------------
def load_drug_documents() -> List[Document]:
    """按二级标题（## 药品名）切分药品说明书，每种药品转成一个 Document。"""
    with open(config.DRUG_MARKDOWN, "r", encoding="utf-8") as f:
        text = f.read()

    documents = []
    # 以“## ”作为每种药品的分隔标志进行切分
    # 使用前瞻正则，保证切分后每段仍以 "## 药品名" 开头
    sections = re.split(r"\n(?=## )", text)
    for section in sections:
        section = section.strip()
        # 跳过一级标题（# 药品说明书库）等非药品段落
        if not section.startswith("## "):
            continue
        # 第一行即药品名称
        first_line = section.splitlines()[0]
        drug_name = first_line.replace("##", "").strip()
        documents.append(
            Document(
                page_content=section,
                metadata={
                    "source": "药品说明书",
                    "category": "drug",
                    "title": drug_name,
                },
            )
        )
    return documents


# ---------------------------------------------------------------------------
# 3. 非结构化数据：医患问答 TXT
# ---------------------------------------------------------------------------
def load_faq_documents() -> List[Document]:
    """按空行切分医患问答，每一组 Q&A 转成一个 Document。"""
    with open(config.FAQ_TEXT, "r", encoding="utf-8") as f:
        text = f.read()

    documents = []
    # 每组问答之间以空行分隔
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    for idx, block in enumerate(blocks, start=1):
        # 提取问题作为标题，便于溯源展示
        question_match = re.search(r"Q[:：]\s*(.+)", block)
        title = question_match.group(1).strip() if question_match else f"问答{idx}"
        documents.append(
            Document(
                page_content=block,
                metadata={
                    "source": "医患问答",
                    "category": "faq",
                    "title": title,
                },
            )
        )
    return documents


def load_all_documents() -> List[Document]:
    """汇总三类数据源的全部 Document。"""
    documents = []
    documents.extend(load_disease_documents())
    documents.extend(load_drug_documents())
    documents.extend(load_faq_documents())
    return documents


def get_knowledge_base_stats() -> dict[str, int]:
    """统计三类知识源当前的文档数量，便于展示扩库后的规模。"""
    disease_count = len(load_disease_documents())
    drug_count = len(load_drug_documents())
    faq_count = len(load_faq_documents())
    return {
        "disease": disease_count,
        "drug": drug_count,
        "faq": faq_count,
        "total": disease_count + drug_count + faq_count,
    }


# ---------------------------------------------------------------------------
# 向量库：构建 / 加载
# ---------------------------------------------------------------------------
def get_embeddings() -> DashScopeEmbeddings:
    """创建通义千问向量化模型实例。"""
    config.check_api_key()
    return DashScopeEmbeddings(model=config.EMBEDDING_MODEL)


def build_vectorstore() -> FAISS:
    """加载全部文档、向量化并构建 FAISS 索引，最后持久化到磁盘。"""
    stats = get_knowledge_base_stats()
    documents = load_all_documents()
    print(
        "[知识库] 数据已加载："
        f"疾病百科 {stats['disease']} 条，"
        f"药品说明书 {stats['drug']} 条，"
        f"医患问答 {stats['faq']} 条，"
        f"合计 {stats['total']} 条文档。"
    )
    print("[知识库] 开始向量化并构建 FAISS 索引……")

    embeddings = get_embeddings()
    # normalize_L2=True：把向量归一化，使 L2 距离等价于余弦距离，
    # 从而可以用统一的余弦相似度阈值进行过滤。
    vectorstore = FAISS.from_documents(
        documents,
        embeddings,
        normalize_L2=True,
    )
    vectorstore.save_local(config.INDEX_DIR)
    print(f"[知识库] FAISS 索引已保存至：{config.INDEX_DIR}")
    return vectorstore


def load_vectorstore() -> FAISS:
    """从磁盘加载已构建好的 FAISS 索引；若不存在则自动构建。"""
    if not os.path.exists(os.path.join(config.INDEX_DIR, "index.faiss")):
        print("[知识库] 未找到已有索引，将自动构建……")
        return build_vectorstore()

    embeddings = get_embeddings()
    # allow_dangerous_deserialization=True：加载本地可信的 pickle 索引所需
    return FAISS.load_local(
        config.INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
