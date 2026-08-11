"""
项目全局配置模块。

集中管理路径、模型名称、检索参数与固定文案，方便统一调整与在 PPT 中讲解技术选型。
"""
import os

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
# 项目根目录（本文件位于 src/ 下，因此上溯一级即为根目录）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 本地知识库数据目录
DATA_DIR = os.path.join(BASE_DIR, "data")

# 三类原始数据文件
DISEASE_JSON = os.path.join(DATA_DIR, "disease_encyclopedia.json")   # 结构化：疾病百科
DRUG_MARKDOWN = os.path.join(DATA_DIR, "drug_manuals.md")            # 半结构化：药品说明书
FAQ_TEXT = os.path.join(DATA_DIR, "patient_faq.txt")                 # 非结构化：医患问答

# FAISS 向量索引持久化目录
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")

# ---------------------------------------------------------------------------
# 模型配置（阿里百炼 / 通义千问）
# ---------------------------------------------------------------------------
# 文本向量化模型（用于把文档与问题转成向量）
EMBEDDING_MODEL = "text-embedding-v2"

# 对话大模型（用于根据检索结果生成回答）
LLM_MODEL = "qwen-plus"

# LLM 温度：医疗场景要求稳定、严谨，设为 0
LLM_TEMPERATURE = 0

# 百炼 API Key 环境变量名（用户已在系统环境变量中配置）
API_KEY_ENV = "DASHSCOPE_API_KEY"

# ---------------------------------------------------------------------------
# 检索参数
# ---------------------------------------------------------------------------
# 每次检索返回的候选文档数量
TOP_K = 4

# 文档检索相似度阈值：余弦相似度低于此值的结果视为不相关，将被过滤掉
SIMILARITY_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# 固定文案
# ---------------------------------------------------------------------------
# 检索工具在“无相关结果”时返回的内部标记
NO_RESULT_FLAG = "NO_RELEVANT_KNOWLEDGE"

# 知识库无结果时，Agent 必须原样输出的拒答话术
REFUSAL_MESSAGE = (
    "抱歉，本地知识库中暂无关于该问题的相关数据，我无法为您提供专业解答。"
    "建议您咨询专业医师或查阅权威医疗资料。"
)


def check_api_key() -> str:
    """校验百炼 API Key 是否已在环境变量中配置，未配置则抛出明确错误。"""
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise EnvironmentError(
            f"未检测到环境变量 {API_KEY_ENV}，请先配置阿里百炼 API Key 后再运行。"
        )
    return api_key
