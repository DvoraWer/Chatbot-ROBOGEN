"""מודל ה-embedding וקידומות המשימה של Gemini Embedding 2.

gemini-embedding-2 לא מקבל task_type; לפי התיעוד מוסיפים קידומת לטקסט:
  קטע מסמך: "title: {title} | text: {content}"
  שאלה:     "task: search result | query: {content}"
format_for_embedding היא הפונקציה היחידה שבונה את הקידומות, וגם ה-ingestion וגם השליפה משתמשים בה.
"""
from typing import List, Optional

from google.genai.types import EmbedContentConfig
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

import config


def format_for_embedding(text: str, kind: str, title: Optional[str] = None) -> str:
    """מוסיף קידומת משימה לטקסט. kind הוא "document" או "query"."""
    if not config.EMBED_TASK_PREFIXES:
        return text
    if kind == "document":
        return f"title: {title or 'none'} | text: {text}"
    if kind == "query":
        return f"task: search result | query: {text}"
    raise ValueError(f"kind לא מוכר: {kind}")


def node_embedding_text(node: BaseNode) -> str:
    """הטקסט שנשלח ל-embedding עבור צ'אנק: קידומת עם title, מטא-דאטה (title_en, products) ותוכן."""
    text = node.get_content(metadata_mode=MetadataMode.EMBED)
    return format_for_embedding(text, "document", title=node.metadata.get("title"))


class PrefixedGeminiEmbedding(GoogleGenAIEmbedding):
    """מוסיף את קידומת השאלה אוטומטית, כך שכל retriever (גם בתוך QueryFusionRetriever) משתמש בה."""

    def _get_query_embedding(self, query: str) -> List[float]:
        return super()._get_query_embedding(format_for_embedding(query, "query"))

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await super()._aget_query_embedding(format_for_embedding(query, "query"))


def get_embed_model(model_class=PrefixedGeminiEmbedding) -> PrefixedGeminiEmbedding:
    """מודל ה-embedding עם ממד 768 ו-retry מובנה (המתנה הולכת וגדלה).
    model_class מאפשר לכלי הבדיקה להחליף במחלקה עם cache."""
    return model_class(
        model_name=config.EMBED_MODEL,
        api_key=config.require_api_key(),
        embed_batch_size=config.EMBED_BATCH_SIZE,
        embedding_config=EmbedContentConfig(output_dimensionality=config.EMBED_DIM),
        retries=config.API_RETRIES,
        retry_min_seconds=config.RETRY_MIN_SECONDS,
        retry_max_seconds=config.RETRY_MAX_SECONDS,
    )


def embed_nodes(nodes: List[BaseNode], embed_model: GoogleGenAIEmbedding) -> None:
    """מחשב embeddings לצ'אנקים (בקבוצות של EMBED_BATCH_SIZE) ושומר אותם על ה-node."""
    texts = [node_embedding_text(node) for node in nodes]
    vectors = embed_model.get_text_embedding_batch(texts, show_progress=True)
    for node, vector in zip(nodes, vectors):
        node.embedding = vector
