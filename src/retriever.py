"""חיפוש היברידי: Vector (Chroma) + BM25, מאוחדים ב-RRF דרך QueryFusionRetriever.

הרצה ישירה (בלי LLM) מדפיסה את הצ'אנקים שנשלפו:
    python src/retriever.py "מה כושר הנשיאה של CARGO?"
    python src/retriever.py "מה זה RG-03?" --mode bm25
"""
import argparse
import math

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

import config
from embedding import get_embed_model

# טוקניזציה ל-BM25: רצף של אותיות/ספרות (בלי קו תחתון). bm25s גם ממיר ל-lowercase.
BM25_TOKEN_PATTERN = r"(?u)[^\W_]+"
HEBREW_PREFIXES = set("הבלומשכ")


def strip_hebrew_prefixes(word: str) -> str:
    """מסיר אותיות תחילית מתחילת מילה עברית, כל עוד נשארות לפחות 3 אותיות.

    ההסרה עקבית: "במחסן", "המחסן" ו-"מחסן" כולן הופכות ל-"חסן", כך שהן מתאימות זו לזו.
    המחיר: לפעמים מילים שונות מתמזגות (למשל "מספר" ו-"ספר"). מילים באנגלית ומספרים לא משתנים.
    """
    while len(word) > 3 and word[0] in HEBREW_PREFIXES:
        word = word[1:]
    return word


def strip_hebrew_prefixes_all(words: list) -> list:
    """bm25s קורא ל-stemmer עם רשימת מילים (לא מילה בודדת), לכן עוטפים."""
    return [strip_hebrew_prefixes(w) for w in words]


def build_vector_retriever(top_k: int):
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = client.get_collection(config.CHROMA_COLLECTION)
    index = VectorStoreIndex.from_vector_store(
        ChromaVectorStore(chroma_collection=collection),
        embed_model=get_embed_model(),  # מוסיף את קידומת השאלה אוטומטית
    )
    return index.as_retriever(similarity_top_k=top_k)


def build_bm25_retriever(top_k: int):
    """BM25 נבנה מה-docstore השמור, בלי embeddings."""
    docstore = SimpleDocumentStore.from_persist_path(str(config.DOCSTORE_DIR / "docstore.json"))
    strip = config.BM25_STRIP_HEBREW_PREFIXES
    return BM25Retriever.from_defaults(
        docstore=docstore,
        similarity_top_k=top_k,
        stemmer=strip_hebrew_prefixes_all if strip else None,
        skip_stemming=not strip,  # בלי זה BM25Retriever משתמש ב-stemmer אנגלי
        token_pattern=BM25_TOKEN_PATTERN,
    )


def build_hybrid_retriever(vector_retriever, bm25_retriever):
    return QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        mode="reciprocal_rerank",           # RRF
        similarity_top_k=config.FINAL_TOP_K,
        num_queries=config.NUM_QUERIES,     # 1 = בלי Query Expansion
        llm=config.get_llm(),               # בשימוש רק כש-num_queries > 1
        use_async=False,
    )


def build_retriever(mode: str = config.RETRIEVAL_MODE):
    """מחזיר retriever לפי מצב: "hybrid", "vector" או "bm25". תמיד FINAL_TOP_K קטעים בסוף."""
    if mode == "vector":
        return build_vector_retriever(config.FINAL_TOP_K)
    if mode == "bm25":
        return build_bm25_retriever(config.FINAL_TOP_K)
    if mode == "hybrid":
        k = config.CANDIDATES_TOP_K  # מועמדים מכל retriever; ה-RRF בוחר מהם FINAL_TOP_K
        return build_hybrid_retriever(build_vector_retriever(k), build_bm25_retriever(k))
    raise ValueError(f"mode לא מוכר: {mode}")


def print_results(question: str, mode: str) -> None:
    print(f"שאלה: {question}\nמצב: {mode}\n")

    if mode == "hybrid":
        # מריצים גם כל retriever בנפרד, כדי להראות מאיפה הגיע כל צ'אנק
        k = config.CANDIDATES_TOP_K
        vector, bm25 = build_vector_retriever(k), build_bm25_retriever(k)
        vector_rank = {r.node.node_id: i for i, r in enumerate(vector.retrieve(question), 1)}
        bm25_rank = {r.node.node_id: i for i, r in enumerate(bm25.retrieve(question), 1)}
        results = build_hybrid_retriever(vector, bm25).retrieve(question)
    else:
        results = build_retriever(mode).retrieve(question)

    for i, r in enumerate(results, 1):
        meta = r.node.metadata
        if mode == "hybrid":
            score = (f"RRF={r.score:.4f} | vector #{vector_rank.get(r.node.node_id, '-')}"
                     f" | bm25 #{bm25_rank.get(r.node.node_id, '-')}")
        elif mode == "vector":
            score = f"cosine={1 + math.log(r.score):.3f}"  # Chroma מחזיר exp(-distance)
        else:
            score = f"bm25={r.score:.3f}"
        snippet = " ".join(r.node.get_content().split())[:160]
        print(f"[{i}] {meta['doc_id']} – {meta['title']} | {score}")
        print(f"    {snippet}...\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="בדיקת שליפה בלי LLM")
    parser.add_argument("question")
    parser.add_argument("--mode", default=config.RETRIEVAL_MODE, choices=["hybrid", "vector", "bm25"])
    args = parser.parse_args()
    print_results(args.question, args.mode)
