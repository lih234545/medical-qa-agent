"""
医疗问诊 AI Agent 统一入口。

用法：
    python main.py cli      # 命令行交互（默认）
    python main.py web      # 启动 Gradio Web 界面
    python main.py build    # 构建 / 重建向量索引

可选参数：
    --verbose               # 命令行模式下打印 Agent 的中间推理过程
    --share                 # Web 模式下生成公网可访问链接
"""
import argparse
import sys

# Windows 控制台默认 GBK 编码，模型回答可能包含 emoji 等字符导致打印报错，
# 这里统一把标准输入输出切换为 UTF-8。
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def main():
    parser = argparse.ArgumentParser(description="医疗问诊 AI Agent")
    parser.add_argument(
        "mode",
        nargs="?",
        default="cli",
        choices=["cli", "web", "build"],
        help="运行模式：cli（命令行）/ web（网页）/ build（构建索引）",
    )
    parser.add_argument("--verbose", action="store_true", help="打印 Agent 中间过程")
    parser.add_argument("--share", action="store_true", help="Web 模式生成公网链接")
    args = parser.parse_args()

    if args.mode == "build":
        from build_index import main as build_main
        build_main()
    elif args.mode == "web":
        from src.web import run_web
        run_web(share=args.share)
    else:
        from src.cli import run_cli
        run_cli(verbose=args.verbose)


if __name__ == "__main__":
    main()
