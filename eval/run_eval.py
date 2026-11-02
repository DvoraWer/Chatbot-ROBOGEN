"""בדיקות: שליפה (בלי LLM) בכמה תצורות, ותשובות (עם LLM) לתצורה הראשית בלבד.

הרצה:
    python eval/run_eval.py            # שליפה + תשובות
    python eval/run_eval.py --no-llm   # שליפה בלבד (חוסך במכסה)
    python eval/run_eval.py --questions eval/questions_holdout.json --tag holdout_after
        # קובץ שאלות אחר; הפלט נשמר כ-results_<tag>_retrieval.csv / results_<tag>_answers.json
    python eval/run_eval.py --configs main --tag final
        # רק חלק מהתצורות (main תמיד נדרשת)

Cache: ה-embeddings של שאלות הבדיקה נשמרים ב-eval/cache/query_embeddings.json
(מפתח: מצב הקידומות + טקסט השאלה), כך שהרצות חוזרות לא צורכות מכסת embedding.
ה-cache קיים רק בכלי הבדיקה; המערכת עצמה (rag.py / app.py) לא משתמשת בו.

מדדי שליפה (על 6 הצ'אנקים שנשלפו; שאלות בלי expected_docs מדולגות):
    Hit@6      – האם לפחות מסמך צפוי אחד נשלף.
    first rank – המיקום (1–6) של הצ'אנק הראשון ממסמך צפוי.
    MRR        – ממוצע של 1/first rank (0 אם לא נשלף).
    Recall@6   – איזה חלק מהמסמכים הצפויים נשלף (חשוב בשאלות שצריכות כמה מסמכים).
"""
import argparse
import contextlib
import csv
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import ingest  # noqa: E402
import retriever  # noqa: E402
from embedding import PrefixedGeminiEmbedding, get_embed_model  # noqa: E402
from rag import RagEngine, is_not_found  # noqa: E402
from retriever import build_retriever  # noqa: E402

EVAL_DIR = ROOT / "eval"
NO_PREFIX_STORAGE = config.STORAGE_DIR / "eval_no_prefixes"
CACHE_FILE = EVAL_DIR / "cache" / "query_embeddings.json"


# ---------- cache ל-embeddings של שאלות (בכלי הבדיקה בלבד) ----------

def load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


QUERY_CACHE = load_cache()
CACHE_STATS = {"hits": 0, "misses": 0}


class CachedQueryEmbedding(PrefixedGeminiEmbedding):
    """כמו מודל ה-embedding של המערכת, אבל שומר וקטורי שאלות בקובץ."""

    def _get_query_embedding(self, query: str):
        key = f"prefixes={config.EMBED_TASK_PREFIXES}|{query}"
        if key in QUERY_CACHE:
            CACHE_STATS["hits"] += 1
            return QUERY_CACHE[key]
        vector = super()._get_query_embedding(query)  # כאן נוספת הקידומת
        CACHE_STATS["misses"] += 1
        QUERY_CACHE[key] = vector
        CACHE_FILE.parent.mkdir(exist_ok=True)
        CACHE_FILE.write_text(json.dumps(QUERY_CACHE), encoding="utf-8")  # שומרים מיד: עמיד לנפילה באמצע
        return vector


def get_cached_embed_model() -> CachedQueryEmbedding:
    return get_embed_model(model_class=CachedQueryEmbedding)


# retriever.py יוצר את מודל ה-embedding דרך get_embed_model; בכלי הבדיקה מחליפים אותו בגרסה עם cache.
retriever.get_embed_model = get_cached_embed_model

# כל תצורה משנה פרמטר אחד ביחס לתצורה הראשית (הערכים ב-config.py).
CONFIGS = {
    "main": {},
    "vector_only": {"RETRIEVAL_MODE": "vector"},
    "bm25_only": {"RETRIEVAL_MODE": "bm25"},
    "no_embed_prefixes": {
        "EMBED_TASK_PREFIXES": False,
        # וקטורים אחרים → אינדקס נפרד, כדי לא לדרוס את הראשי
        "STORAGE_DIR": NO_PREFIX_STORAGE,
        "CHROMA_DIR": NO_PREFIX_STORAGE / "chroma",
        "DOCSTORE_DIR": NO_PREFIX_STORAGE / "docstore",
    },
    "no_hebrew_prefix_strip": {"BM25_STRIP_HEBREW_PREFIXES": False},
    # להשוואה מול bm25_only: ב-hybrid ההשפעה של הסרת התחיליות מוסתרת
    "bm25_no_strip": {"RETRIEVAL_MODE": "bm25", "BM25_STRIP_HEBREW_PREFIXES": False},
    # מספר המועמדים ל-RRF: 6 הוא הערך הקודם (לפני תיקון A), 8 ערך ביניים להשוואה בלבד
    "fusion_candidates_6": {"CANDIDATES_TOP_K": 6},
    "fusion_candidates_8": {"CANDIDATES_TOP_K": 8},
    # Query Expansion (D): המודל מייצר 2 ניסוחים נוספים; קריאת LLM לכל שאלה
    "query_expansion_3": {"NUM_QUERIES": 3},
}


@contextlib.contextmanager
def override(params: dict):
    """משנה זמנית ערכים ב-config, ומחזיר אותם בסוף."""
    old = {key: getattr(config, key) for key in params}
    for key, value in params.items():
        setattr(config, key, value)
    try:
        yield
    finally:
        for key, value in old.items():
            setattr(config, key, value)


def doc_ids(nodes) -> list:
    return [n.node.metadata["doc_id"] for n in nodes]


def first_relevant_rank(retrieved_docs: list, expected: list):
    for rank, doc_id in enumerate(retrieved_docs, 1):
        if doc_id in expected:
            return rank
    return None


# ---------- שליפה ----------

def evaluate_retrieval(name: str, params: dict, questions: list, rebuild_variants: bool) -> list:
    with override(params):
        # אינדקס נפרד נבנה רק אם חסר (או ב---rebuild-variants), כדי לא לצרוך מכסת embedding בכל הרצה
        if "STORAGE_DIR" in params and (rebuild_variants or not (config.DOCSTORE_DIR / "docstore.json").exists()):
            print(f"  בונה אינדקס נפרד עבור {name}...")
            with contextlib.redirect_stdout(io.StringIO()):  # בלי הדפסות ה-ingestion
                ingest.main()

        ret = build_retriever(config.RETRIEVAL_MODE)
        rows = []
        for qid, q in enumerate(questions, 1):
            if not q["expected_docs"]:
                continue
            misses_before = CACHE_STATS["misses"]
            retrieved = doc_ids(ret.retrieve(q["question"]))
            rank = first_relevant_rank(retrieved, q["expected_docs"])
            found = set(q["expected_docs"]) & set(retrieved)
            rows.append({
                "config": name, "id": qid, "type": q["type"], "question": q["question"],
                "expected_docs": " ".join(q["expected_docs"]),
                "hit": int(rank is not None), "first_rank": rank or "",
                "recall": round(len(found) / len(q["expected_docs"]), 3),
                "retrieved_docs": " ".join(retrieved),
            })
            if config.NUM_QUERIES > 1:  # Query Expansion קורא ל-LLM
                time.sleep(config.EVAL_LLM_DELAY_SECONDS)
            elif CACHE_STATS["misses"] > misses_before:  # המתנה רק אם הייתה קריאת API
                time.sleep(config.EVAL_EMBED_DELAY_SECONDS)
    return rows


def summarize(rows: list) -> dict:
    ranks = [r["first_rank"] or None for r in rows]
    hits = [r for r in ranks if r]
    return {
        "hit": sum(1 for r in ranks if r) / len(ranks),
        "mrr": sum(1 / r for r in hits) / len(ranks),
        "avg_rank": sum(hits) / len(hits) if hits else float("nan"),
        "recall": sum(float(r["recall"]) for r in rows) / len(rows),
    }


def print_retrieval_tables(all_rows: dict) -> None:
    print("\n===== סיכום שליפה =====")
    print(f"{'config':<24}{'Hit@6':>8}{'MRR':>8}{'Recall@6':>10}{'avg first rank':>16}")
    for name, rows in all_rows.items():
        s = summarize(rows)
        print(f"{name:<24}{s['hit']:>8.2f}{s['mrr']:>8.3f}{s['recall']:>10.3f}{s['avg_rank']:>16.2f}")

    print("\n===== מיקום המסמך הנכון הראשון, לפי שאלה ('-' = לא נשלף) =====")
    names = list(all_rows)
    print(f"{'#':<4}{'type':<16}" + "".join(f"{n[:12]:>14}" for n in names))
    for i, row in enumerate(all_rows["main"]):
        cells = "".join(f"{(all_rows[n][i]['first_rank'] or '-'):>14}" for n in names)
        print(f"{row['id']:<4}{row['type']:<16}" + cells)


# ---------- תשובות ----------

def evaluate_answers(questions: list) -> list:
    engine = RagEngine(config.RETRIEVAL_MODE)
    results = []
    for qid, q in enumerate(questions, 1):
        print(f"  שאלה {qid}/{len(questions)}...")
        r = engine.query(q["question"])
        cited_docs = [doc_id for doc_id, _ in r.sources]
        expected = q["expected_docs"]
        results.append({
            "id": qid,
            "type": q["type"],
            "used_for_diagnosis": q.get("used_for_diagnosis", False),
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "answer": r.answer,
            "says_not_found": is_not_found(r.answer),
            "citations_parsed": r.citations_ok,
            "cited_chunks": r.cited_chunks,
            "sources": cited_docs,
            "expected_docs": expected,
            "sources_in_expected": [d for d in cited_docs if d in expected],
            "sources_not_in_expected": [d for d in cited_docs if d not in expected],
            "retrieved_docs": doc_ids(r.source_nodes),
            # לבדיקה ידנית:
            "manual_correct": None,
            "manual_hallucination": None,
            "manual_notes": "",
        })
        time.sleep(config.EVAL_LLM_DELAY_SECONDS)
    return results


def print_answer_table(results: list) -> None:
    print("\n===== סיכום תשובות (תצורה ראשית) =====")
    print(f"{'#':<4}{'type':<16}{'parsed':>8}{'not_found':>11}  sources (✓ בתוך expected_docs, ✗ מחוץ)")
    for r in results:
        marks = [f"{d}{'✓' if d in r['expected_docs'] else '✗'}" for d in r["sources"]]
        print(f"{r['id']:<4}{r['type']:<16}{str(r['citations_parsed']):>8}"
              f"{str(r['says_not_found']):>11}  {' '.join(marks) or '-'}")


def output_path(kind: str, tag: str) -> Path:
    """results_<kind> או results_<tag>_<kind>."""
    return EVAL_DIR / (f"results_{tag}_{kind}" if tag else f"results_{kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", help="שליפה בלבד, בלי יצירת תשובות")
    parser.add_argument("--questions", default=str(EVAL_DIR / "questions.json"), help="קובץ השאלות")
    parser.add_argument("--tag", default="", help="תוספת לשמות קבצי הפלט, למשל before / holdout_after")
    parser.add_argument("--configs", default=",".join(CONFIGS),
                        help="תצורות להרצה, מופרדות בפסיק (main תמיד נדרשת). ברירת מחדל: כולן")
    parser.add_argument("--rebuild-variants", action="store_true",
                        help="לבנות מחדש אינדקסים נפרדים (למשל no_embed_prefixes) גם אם קיימים")
    args = parser.parse_args()

    selected = [name.strip() for name in args.configs.split(",") if name.strip()]
    unknown = [name for name in selected if name not in CONFIGS]
    if unknown or "main" not in selected:
        sys.exit(f"--configs חייב לכלול main ותצורות מתוך: {', '.join(CONFIGS)} (לא מוכר: {unknown})")

    if not (config.DOCSTORE_DIR / "docstore.json").exists():
        sys.exit("לא נמצא אינדקס. הריצו קודם: python src/ingest.py")

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    print(f"קובץ שאלות: {args.questions} ({len(questions)} שאלות)")

    print("בדיקת שליפה...")
    all_rows = {}
    for name in selected:
        print(f"  {name}")
        all_rows[name] = evaluate_retrieval(name, CONFIGS[name], questions, args.rebuild_variants)

    retrieval_file = output_path("retrieval.csv", args.tag)
    with open(retrieval_file, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig: עברית תקינה ב-Excel
        rows = [row for rows in all_rows.values() for row in rows]
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print_retrieval_tables(all_rows)
    print(f"\nנשמר: {retrieval_file}")
    print(f"cache של embeddings: {CACHE_STATS['hits']} מה-cache, {CACHE_STATS['misses']} קריאות API")

    if args.no_llm:
        return

    print("\nיצירת תשובות (תצורה ראשית)...")
    results = evaluate_answers(questions)
    answers_file = output_path("answers.json", args.tag)
    answers_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print_answer_table(results)
    print(f"\nנשמר: {answers_file}")


if __name__ == "__main__":
    main()
