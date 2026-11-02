"""RAG: שליפה היברידית + יצירת תשובה עם Gemini + הצגת מקורות.

הרצה מהטרמינל:
    python src/rag.py "מה כושר הנשיאה של CARGO?"
    python src/rag.py "שאלה 1" "שאלה 2" --show-prompt --show-chunks
    python src/rag.py "מה כושר הנשיאה של CARGO?" "ומה לגבי SCOUT?" --chat   # שיחה עם זיכרון
"""
import argparse
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events.llm import LLMChatStartEvent
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.schema import MetadataMode, NodeWithScore

import config
from retriever import build_retriever

NOT_FOUND = "המידע אינו מופיע במסמכי החברה"

SYSTEM_PROMPT = f"""אתה עוזר המידע של חברת ROBOGEN. תפקידך לענות על שאלות על סמך קטעים ממסמכי החברה.

כללים:
1. ענה רק על סמך הקטעים שסופקו. אל תשתמש בידע כללי ואל תשלים פרטים שאינם מופיעים בקטעים.
2. אם המידע הדרוש אינו מופיע בקטעים, כתוב במפורש: "{NOT_FOUND}", ואל תנחש. אם השאלה כוללת כמה חלקים ורק חלקם מתועדים, ענה על החלקים המתועדים וציין איזה חלק מהשאלה לא נענה. אל תוסיף הערות על מידע שלא נשאל עליו.
3. בשאלת השוואה או דירוג (למשל "איזה רובוט הכי..."): הצג את הנתון המתועד לכל פריט שמופיע בקטעים, וציין מי מוביל מבין הפריטים שהנתון שלהם מתועד. ציין במפורש לאילו פריטים הנתון חסר. אל תציג את המוביל כמסקנה על כלל המוצרים, ואם הנתון מתועד לפריט אחד בלבד – אל תקבע מוביל.
4. אם השאלה מניחה משהו שסותר את המסמכים, אל תקבל את ההנחה: ענה לפי הנתון המתועד והסבר את הפער.
5. אם יש הבדל בין מסמך כללי לבין מפרט טכני של מוצר, העדף את המפרט הטכני.
6. ענה בעברית. שמור שמות מוצרים ומונחים טכניים באנגלית (למשל CARGO, LiDAR).
7. ענה בתמציתיות. בשאלות השוואה מותר להשתמש ברשימה.
8. בסוף התשובה, בשורה נפרדת, ציין את מספרי הקטעים שעליהם התבססת, בדיוק בפורמט: SOURCES: [1, 3]
   אם המידע אינו מופיע בקטעים, כתוב: SOURCES: []

כל קטע מסומן [קטע N] ומתחיל בשורות doc_id, title ו-products, שמזהות את המסמך שממנו הוא נלקח."""

USER_TEMPLATE = """קטעים ממסמכי החברה:
---------------------
{context_str}
---------------------

שאלה: {query_str}"""

QA_TEMPLATE = ChatPromptTemplate([
    ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
    ChatMessage(role=MessageRole.USER, content=USER_TEMPLATE),
])

# "SOURCES: [1, 3]" (סובלני להדגשת markdown כמו **SOURCES:**)
SOURCES_PATTERN = re.compile(r"\**SOURCES\**:\**\s*\[([\d,\s]*)\]\**")
PARSE_FAILED_NOTE = "לא ניתן היה לזהות את הקטעים שהמודל ציין; מוצגים כל המסמכים שנשלפו."

# ניסוח מחדש של שאלת המשך לשאלה עצמאית (זיכרון שיחה)
CONDENSE_PROMPT = """נתונה היסטוריית שיחה עם עוזר המידע של ROBOGEN, ושאלת המשך.
נסח מחדש את שאלת ההמשך כשאלה עצמאית ומלאה, שמובנת בלי ההיסטוריה: החלף כינויים והפניות ("הוא", "מהם", "ומה לגבי...") בשמות המוצרים והנושאים שאליהם הם מתייחסים.
אם השאלה כבר עצמאית – החזר אותה בדיוק כמו שהיא.
אל תענה על השאלה ואל תוסיף מידע. החזר רק את השאלה המנוסחת, בשורה אחת, בשפה של שאלת ההמשך.

היסטוריה:
{history}

שאלת המשך: {question}
שאלה עצמאית:"""

# סבב שיחה: (השאלה שעליה נענה – אחרי ניסוח מחדש, התשובה)
Turn = Tuple[str, str]


@dataclass
class RagResult:
    question: str                        # השאלה כפי שהמשתמש כתב
    search_question: str                 # השאלה שעליה נעשו החיפוש והתשובה (אחרי ניסוח מחדש)
    answer: str                          # התשובה בלי שורת SOURCES
    sources: List[Tuple[str, str]]       # (doc_id, title) להצגה, בלי כפילויות
    cited_chunks: Optional[List[int]]    # מספרי הקטעים שצוינו; None = הפענוח נכשל
    source_nodes: List[NodeWithScore]    # כל הקטעים שנשלפו (לתצוגת בדיקה)

    @property
    def citations_ok(self) -> bool:
        return self.cited_chunks is not None

    @property
    def was_rewritten(self) -> bool:
        return self.search_question.strip() != self.question.strip()


def format_context(nodes: List[NodeWithScore]) -> str:
    """הקטעים ממוספרים, כל אחד עם המטא-דאטה שהמודל רואה (title, doc_id, products)."""
    return "\n\n".join(
        f"[קטע {i}]\n{n.node.get_content(metadata_mode=MetadataMode.LLM)}"
        for i, n in enumerate(nodes, 1)
    )


def parse_citations(raw_answer: str, num_chunks: int) -> Tuple[str, Optional[List[int]]]:
    """מפריד את שורת SOURCES מהתשובה. מחזיר (תשובה, מספרי קטעים או None אם הפענוח נכשל)."""
    matches = list(SOURCES_PATTERN.finditer(raw_answer))
    if not matches:
        return raw_answer.strip(), None
    last = matches[-1]
    answer = (raw_answer[:last.start()] + raw_answer[last.end():]).strip()
    numbers = [int(x) for x in re.findall(r"\d+", last.group(1))]
    if any(n < 1 or n > num_chunks for n in numbers):
        return answer, None
    return answer, list(dict.fromkeys(numbers))  # בלי כפילויות, לפי הסדר


def unique_sources(nodes: List[NodeWithScore]) -> List[Tuple[str, str]]:
    """(doc_id, title) לכל מסמך, לפי הסדר, בלי כפילויות."""
    sources = {}
    for n in nodes:
        sources.setdefault(n.node.metadata["doc_id"], n.node.metadata["title"])
    return list(sources.items())


def is_not_found(answer: str) -> bool:
    return answer.strip().rstrip(".") == NOT_FOUND


class RagEngine:
    """שליפה (6 צ'אנקים) ← בניית הפרומפט ← קריאה אחת למודל ← פענוח המקורות.

    בונים את הקונטקסט בעצמנו במקום response synthesizer של LlamaIndex:
    כך הפורמט גלוי ופשוט (ו-simple_summarize מעביר את הקטעים כרשימת Python).
    """

    def __init__(self, mode: str = config.RETRIEVAL_MODE):
        self.retriever = build_retriever(mode)
        self.llm = config.get_llm()

    def condense_question(self, question: str, history: List[Turn]) -> str:
        """שאלת המשך ← שאלה עצמאית, על סמך MEMORY_TURNS הסבבים האחרונים."""
        recent = history[-config.MEMORY_TURNS:]
        history_text = "\n\n".join(f"משתמש: {q}\nעוזר: {a}" for q, a in recent)
        prompt = CONDENSE_PROMPT.format(history=history_text, question=question)
        rewritten = self.llm.chat([ChatMessage(role=MessageRole.USER, content=prompt)]).message.content
        lines = [line.strip().strip('"') for line in rewritten.splitlines() if line.strip()]
        return lines[0] if lines else question  # תשובה ריקה ← השאלה המקורית

    def query(self, question: str, history: Optional[List[Turn]] = None) -> RagResult:
        """history: סבבים קודמים בשיחה. בלי היסטוריה (או כשהזיכרון כבוי) – אין ניסוח מחדש."""
        search_question = question
        if config.CONVERSATION_MEMORY and history:
            search_question = self.condense_question(question, history)

        nodes = self.retriever.retrieve(search_question)
        messages = QA_TEMPLATE.format_messages(context_str=format_context(nodes), query_str=search_question)
        raw_answer = self.llm.chat(messages).message.content
        answer, cited = parse_citations(raw_answer, len(nodes))

        if is_not_found(answer):
            sources = []                                            # אין מידע ← אין מקורות
        elif cited is None:
            sources = unique_sources(nodes)                         # הפענוח נכשל ← כל מה שנשלף
        else:
            sources = unique_sources([nodes[i - 1] for i in cited])  # רק הקטעים שצוינו
        return RagResult(question, search_question, answer, sources, cited, nodes)


class PromptPrinter(BaseEventHandler):
    """מדפיס את ההודעות בדיוק כפי שהן נשלחות למודל (לצורך בדיקה)."""

    @classmethod
    def class_name(cls) -> str:
        return "PromptPrinter"

    def handle(self, event, **kwargs) -> None:
        if isinstance(event, LLMChatStartEvent):
            print("=" * 25, "הפרומפט המלא שנשלח למודל", "=" * 25)
            for message in event.messages:
                print(f"----- [{message.role.value}] -----\n{message.content}\n")
            print("=" * 75)


def main() -> None:
    parser = argparse.ArgumentParser(description="שאלה ← תשובה + מקורות")
    parser.add_argument("questions", nargs="+")
    parser.add_argument("--show-prompt", action="store_true", help="הדפסת הפרומפט המלא")
    parser.add_argument("--show-chunks", action="store_true", help="הדפסת כל הקטעים שנשלפו")
    parser.add_argument("--chat", action="store_true",
                        help="השאלות הן שיחה אחת (עם זיכרון), ולא שאלות נפרדות")
    args = parser.parse_args()

    if args.show_prompt:
        get_dispatcher().add_event_handler(PromptPrinter())

    engine = RagEngine()
    history: List[Turn] = []
    for question in args.questions:
        result = engine.query(question, history if args.chat else None)
        history.append((result.search_question, result.answer))
        print(f"\nשאלה: {question}")
        if result.was_rewritten:
            print(f"חיפשתי: {result.search_question}")
        print(f"תשובה:\n{result.answer}")
        print(f"קטעים שצוינו: {result.cited_chunks}")
        print("מקורות:" if result.sources else "מקורות: אין")
        if not result.citations_ok:
            print(f"  ({PARSE_FAILED_NOTE})")
        for doc_id, title in result.sources:
            print(f"  - {title} ({doc_id})")
        if args.show_chunks:
            for i, n in enumerate(result.source_nodes, 1):
                print(f"  [{i}] {n.node.metadata['doc_id']}: {' '.join(n.node.text.split())[:150]}...")
        print("-" * 60)


if __name__ == "__main__":
    main()
