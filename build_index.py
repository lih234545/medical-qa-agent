"""
向量索引构建脚本。

首次使用或数据更新后运行本脚本，即可重新构建 FAISS 向量索引并保存到 data/faiss_index。

运行方式（在 python-demo 环境下）：
    python build_index.py
"""
from src.knowledge_base import build_vectorstore


def main():
    print("=" * 60)
    print("开始构建医疗知识库向量索引")
    print("=" * 60)
    vectorstore = build_vectorstore()
    total = vectorstore.index.ntotal
    print(f"构建完成，向量总数：{total}")


if __name__ == "__main__":
    main()
