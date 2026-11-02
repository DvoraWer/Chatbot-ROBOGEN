"""Ingestion: טעינת המסמכים, חלוקה לצ'אנקים, embeddings ושמירה ב-ChromaDB + docstore.

הרצה חוזרת בונה הכול מחדש (ה-collection נמחק), כך שאין כפילויות.
"""
import json
from pathlib import Path

import chromadb
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.utils import get_tokenizer
from llama_index.vector_stores.chroma import ChromaVectorStore

import config
from embedding import embed_nodes, get_embed_model, node_embedding_text

# מטא-דאטה שהמודל רואה בזמן יצירת התשובה
LLM_METADATA_KEYS = {"title", "doc_id", "products"}


def embed_metadata_keys() -> set:
    """מטא-דאטה שנכנסת לטקסט ה-embedding.
    כשהקידומות פעילות, title כבר נמצא בקידומת "title: ..." ולכן לא חוזר גם במטא-דאטה."""
    return {"title_en", "products"} | (set() if config.EMBED_TASK_PREFIXES else {"title"})


def load_documents():
    """טוען את 15 קבצי ה-md ומצמיד לכל אחד את המטא-דאטה שלו מ-metadata.json."""
    metadata_by_file = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))

    def file_metadata(file_path: str) -> dict:
        file_name = Path(file_path).name
        return {**metadata_by_file[file_name], "file_name": file_name}

    reader = SimpleDirectoryReader(
        input_dir=str(config.DOCS_DIR),
        required_exts=[".md"],
        file_metadata=file_metadata,
        encoding="utf-8",
    )
    documents = reader.load_data()

    # כל מפתח שלא ברשימות למעלה מוחרג מה-embedding / מה-LLM
    for doc in documents:
        keys = set(doc.metadata)
        doc.excluded_embed_metadata_keys = sorted(keys - embed_metadata_keys())
        doc.excluded_llm_metadata_keys = sorted(keys - LLM_METADATA_KEYS)
    return documents


def split_documents(documents):
    """חלוקה לצ'אנקים; כל צ'אנק יורש את המטא-דאטה ואת רשימות ההחרגה של המסמך."""
    splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    return splitter.get_nodes_from_documents(documents)


def build_chroma_store() -> ChromaVectorStore:
    """מוחק את ה-collection (אם קיים) ויוצר אותו מחדש עם מדד cosine."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if config.CHROMA_COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(config.CHROMA_COLLECTION)
    collection = client.create_collection(
        name=config.CHROMA_COLLECTION,
        configuration={"hnsw": {"space": "cosine"}},  # ברירת המחדל של Chroma היא L2
        embedding_function=None,                       # ה-embeddings מגיעים מ-Gemini
    )
    return ChromaVectorStore(chroma_collection=collection)


def save_docstore(nodes) -> None:
    """שומר את הצ'אנקים לדיסק, כדי ש-BM25 ייבנה בלי לחשב embeddings מחדש."""
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    docstore.persist(str(config.DOCSTORE_DIR / "docstore.json"))


def print_stats(documents, nodes) -> None:
    tokenizer = get_tokenizer()
    avg_chars = sum(len(n.text) for n in nodes) / len(nodes)
    avg_tokens = sum(len(tokenizer(n.text)) for n in nodes) / len(nodes)

    print("\n===== סטטיסטיקות =====")
    print(f"מסמכים: {len(documents)}")
    print(f"צ'אנקים: {len(nodes)}")
    print(f"גודל ממוצע לצ'אנק: {avg_chars:.0f} תווים, {avg_tokens:.0f} טוקנים (תוכן בלבד, ללא מטא-דאטה)")
    print(f"קידומות משימה ל-embedding: {'פעיל' if config.EMBED_TASK_PREFIXES else 'כבוי'}")

    sample = nodes[0]
    print("\n===== דוגמה לצ'אנק =====")
    print("מטא-דאטה מלאה:", json.dumps(sample.metadata, ensure_ascii=False))
    print("\n--- הטקסט שנשלח ל-embedding ---")
    print(node_embedding_text(sample))
    print("\n--- מה שהמודל יראה בזמן יצירת התשובה ---")
    print(sample.get_content(metadata_mode=MetadataMode.LLM))


def main() -> None:
    documents = load_documents()
    print(f"נטענו {len(documents)} מסמכים")

    nodes = split_documents(documents)
    print(f"נוצרו {len(nodes)} צ'אנקים, מחשב embeddings...")

    embed_nodes(nodes, get_embed_model())

    vector_store = build_chroma_store()
    vector_store.add(nodes)
    save_docstore(nodes)
    print(f"נשמר ל-{config.STORAGE_DIR}")

    print_stats(documents, nodes)


if __name__ == "__main__":
    main()
