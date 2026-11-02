"""כל הפרמטרים של הפרויקט במקום אחד."""
import os
from pathlib import Path

import truststore
from dotenv import load_dotenv

# שימוש במאגר התעודות של Windows (נדרש כשסינון כמו NetFree מבצע בדיקת HTTPS).
truststore.inject_into_ssl()

# --- נתיבים ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "data" / "docs"
METADATA_FILE = DOCS_DIR / "metadata.json"
STORAGE_DIR = ROOT_DIR / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
DOCSTORE_DIR = STORAGE_DIR / "docstore"

# --- מפתח API (נטען מ-.env, לא מודפס לעולם) ---
load_dotenv(ROOT_DIR / ".env")
# שם המשתנה ב-.env: GEMINI_API_KEY (רק הוא נתמך).
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- מודלים (מזהים נבדקו מול ai.google.dev, נובמבר 2026) ---
EMBED_MODEL = "gemini-embedding-2"
EMBED_DIM = 768
# קידומות משימה לפי התיעוד של gemini-embedding-2 (ראו embedding.py).
# שינוי הערך מחייב הרצה מחדש של ingest.py, כי הוא משנה את וקטורי הצ'אנקים.
EMBED_TASK_PREFIXES = True
LLM_MODEL = "gemini-3.1-flash-lite"
LLM_TEMPERATURE = 0.1

# --- הגבלת קצב (שכבה חינמית) ---
EMBED_BATCH_SIZE = 10      # כמה טקסטים בכל בקשת embedding
API_RETRIES = 6            # ניסיונות חוזרים בשגיאות 429/5xx
RETRY_MIN_SECONDS = 2      # המתנה ראשונה; גדלה פי 2 בכל ניסיון
RETRY_MAX_SECONDS = 60
EVAL_EMBED_DELAY_SECONDS = 1.0  # המתנה בין שאלות בבדיקת שליפה (קריאת embedding אחת לשאלה)
EVAL_LLM_DELAY_SECONDS = 5.0    # המתנה בין שאלות ביצירת תשובות (~12 קריאות LLM לדקה)

# --- חלוקה (Chunking) ---
# הועלה מ-400/50: ה-tokenizer סופר עברית ביוקר, וצ'אנקים של 400 יצאו קטנים מדי (~5 למסמך).
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80

# --- Vector Store ---
CHROMA_COLLECTION = "robogen"

# --- שליפה ---
# שני ערכי top_k:
# CANDIDATES_TOP_K – כמה קטעים כל retriever (vector, BM25) מעביר לאיחוד ב-RRF. לא נשלחים למודל.
#   הועלה מ-6 ל-10: עם 6, קטעים במקום 7–10 בשתי הרשימות לא הגיעו לאיחוד (אבחון "איזה רובוט הכי מהיר?").
# FINAL_TOP_K – כמה קטעים נשארים אחרי האיחוד ונשלחים למודל (גם במצבי vector/bm25 בלבד).
CANDIDATES_TOP_K = 10
FINAL_TOP_K = 6
NUM_QUERIES = 1            # 1 = ללא Query Expansion
RETRIEVAL_MODE = "hybrid"  # "hybrid" / "vector" / "bm25" (להשוואה בשלב הבדיקות)
BM25_STRIP_HEBREW_PREFIXES = True  # הסרת תחיליות ה,ב,ל,ו,מ,ש,כ (ראו retriever.py)

# --- זיכרון שיחה ---
# כשפעיל ויש היסטוריה: ה-LLM מנסח מחדש את שאלת ההמשך לשאלה עצמאית לפני השליפה.
CONVERSATION_MEMORY = True
MEMORY_TURNS = 3           # כמה סבבים אחרונים (שאלה + תשובה) נשלחים לניסוח מחדש


def require_api_key() -> str:
    """מחזיר את המפתח, או עוצר עם הודעה ברורה אם הוא חסר."""
    if not GEMINI_API_KEY:
        raise RuntimeError("לא נמצא מפתח API. הוסיפו GEMINI_API_KEY לקובץ .env")
    return GEMINI_API_KEY


def get_llm():
    """מודל התשובות עם temperature נמוכה."""
    from google.genai.types import GenerateContentConfig
    from llama_index.llms.google_genai import GoogleGenAI

    return GoogleGenAI(
        model=LLM_MODEL,
        api_key=require_api_key(),
        generation_config=GenerateContentConfig(temperature=LLM_TEMPERATURE),
    )
