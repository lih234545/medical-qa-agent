"""
Gradio Web 交互模块。

启动一个网页版聊天界面，与命令行共用同一套 Agent 逻辑。
"""
import gradio as gr

from src.agent import ask
from src.knowledge_base import get_knowledge_base_stats

# 示例问题：方便演示与 PPT 展示
EXAMPLES = [
    "二甲双胍漏服了要补吃双倍吗？",
    "哮喘突然发作喘不上气，沙丁胺醇气雾剂要喷几下？",
    "痛风正在发作，这时候能开始吃别嘌醇吗？",
    "甲减在吃左甲状腺素钠片，为什么一定要空腹吃？",
    "血脂高平时饮食要注意什么？",
    "今天天气怎么样？",  # 知识库外问题，用于演示拒答
]


def respond(message, history):
    """Gradio ChatInterface 回调：接收用户消息，返回 Agent 回答。"""
    if not message or not message.strip():
        return "请输入您想咨询的健康问题。"
    try:
        return ask(message)
    except Exception as e:  # noqa: BLE001
        return f"抱歉，服务出现异常：{e}"


def build_demo():
    """构建 Gradio 界面。"""
    stats = get_knowledge_base_stats()
    return gr.ChatInterface(
        fn=respond,
        title="🏥 医疗问诊 AI 助手",
        description=(
            "基于本地医疗知识库（疾病百科 / 药品说明书 / 医患问答）的智能问诊助手。"
            f"当前已接入 {stats['total']} 条知识文档。"
            "仅依据知识库内容回答，知识库无相关内容时将礼貌拒答。"
        ),
        examples=EXAMPLES,
    )


def run_web(share: bool = False):
    demo = build_demo()
    demo.launch(share=share)


if __name__ == "__main__":
    run_web()
