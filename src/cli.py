"""
命令行交互模块。

在终端与医疗问诊 Agent 进行多轮问答，输入 exit / quit / 退出 结束会话。
"""
from src.agent import ask


def run_cli(verbose: bool = False):
    print("=" * 60)
    print("医疗问诊 AI 助手（命令行模式）")
    print("提示：本助手仅依据本地知识库回答，输入 exit / 退出 结束。")
    print("=" * 60)

    while True:
        try:
            question = input("\n您想咨询什么？> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见，祝您健康！")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"} or question in {"退出", "再见"}:
            print("再见，祝您健康！")
            break

        try:
            answer = ask(question, verbose=verbose)
        except Exception as e:  # noqa: BLE001
            print(f"\n[出错] {e}")
            continue

        print(f"\n医生助手：{answer}")


if __name__ == "__main__":
    run_cli()
