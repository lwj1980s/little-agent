from pathlib import Path
import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from app.core.llm import embeddings
from app.core.config import settings


def ingest_documents():

    knowledge_dir = Path("./data/knowledge")

    documents = []

    # 读取 Markdown 和 TXT 文件
    for file in knowledge_dir.rglob("*"):

        if file.suffix.lower() not in [".md", ".txt"]:
            continue

        text = file.read_text(encoding="utf-8")

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(file)
                }
            )
        )

    if not documents:
        raise ValueError("No knowledge documents found")

    # 文本切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    chunks = splitter.split_documents(documents)

    # 创建向量数据库
    vectorstore = Chroma(
        collection_name="enterprise_knowledge",
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PATH
    )

    # 使用稳定 ID，避免重复索引
    ids = [
        hashlib.sha256(
            (
                doc.metadata["source"]
                + doc.page_content
            ).encode("utf-8")
        ).hexdigest()
        for doc in chunks
    ]

    vectorstore.add_documents(
        documents=chunks,
        ids=ids
    )

    print(f"Indexed {len(chunks)} chunks")


if __name__ == "__main__":
    ingest_documents()