"""ממשק Streamlit לצ'אטבוט ROBOGEN.

הרצה:
    streamlit run src/app.py
"""
import html
import logging

import httpx
import streamlit as st
import tenacity
from google.genai.errors import APIError

import config
from rag import RagEngine

logger = logging.getLogger(__name__)

SAMPLE_QUESTIONS = [
    "מה כושר הנשיאה המרבי של CARGO?",
    "מה ההבדל בין החיישנים של ATLAS ושל SCOUT?",
    "איזה רובוט מתאים לסיורים במפעל ולמדידת טמפרטורה?",
    "מה עושים אם CARGO לא חוזר לעמדת הטעינה?",
    'האם ATLAS יכול לשאת משקל של 50 ק"ג?',
    "מה המחיר של TITAN?",
]

# כיוון RTL לכל הממשק; קוד נשאר LTR
RTL_CSS = """
<style>
.stApp, [data-testid="stSidebar"], [data-testid="stChatMessage"],
[data-testid="stMarkdownContainer"], [data-testid="stExpander"], [data-testid="stCaptionContainer"] {
    direction: rtl;
    text-align: right;
}
[data-testid="stChatInput"] textarea { direction: rtl; text-align: right; }
code, pre { direction: ltr; text-align: left; }
.chunk {
    white-space: pre-wrap;
    font-size: 0.85rem;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.75rem;
    border-right: 3px solid #999;
    background: rgba(128, 128, 128, 0.08);
}
</style>
"""


@st.cache_resource(show_spinner="טוען את המערכת...")
def load_engine() -> RagEngine:
    """נטען פעם אחת לכל הרצת השרת (Chroma, BM25, מודלים)."""
    return RagEngine()


def friendly_error(error: Exception) -> str:
    """הודעה ידידותית למשתמש, בלי פרטים טכניים."""
    if isinstance(error, tenacity.RetryError):  # retry שמיצה את הניסיונות
        error = error.last_attempt.exception()
    if isinstance(error, APIError):
        if error.code == 429:
            return "המערכת עמוסה כרגע (הגענו למגבלת הבקשות). נסו שוב בעוד כדקה."
        if error.code and error.code >= 500:
            return "שירות המודל אינו זמין כרגע. נסו שוב בעוד כמה רגעים."
    if isinstance(error, (httpx.ConnectError, httpx.TimeoutException, ConnectionError)):
        return "בעיית חיבור לשירות. בדקו את החיבור לאינטרנט ונסו שוב."
    return "אירעה שגיאה בעיבוד השאלה. נסו שוב."


def conversation_history(messages: list) -> list:
    """סבבים קודמים (שאלה שעליה נענה, תשובה) מתוך ההיסטוריה שעל המסך. תשובות שגיאה לא נכללות."""
    return [
        (message["search_question"], message["content"])
        for message in messages
        if message["role"] == "assistant" and "error" not in message
    ]


def answer_question(engine: RagEngine, question: str, history: list) -> dict:
    """מריץ שאלה ומחזיר הודעה לשמירה בהיסטוריה (נתונים פשוטים בלבד)."""
    try:
        result = engine.query(question, history)
    except Exception as error:
        # ללוג רק סוג השגיאה והקוד, לא את ההודעה המלאה
        logger.error("query failed: %s (code=%s)", type(error).__name__, getattr(error, "code", None))
        return {"role": "assistant", "error": friendly_error(error)}

    return {
        "role": "assistant",
        "search_question": result.search_question,
        "rewritten": result.was_rewritten,
        "content": result.answer,
        "sources": result.sources,
        "citations_ok": result.citations_ok,
        "chunks": [
            {
                "doc_id": n.node.metadata["doc_id"],
                "title": n.node.metadata["title"],
                "score": n.score,
                "text": n.node.text,
            }
            for n in result.source_nodes
        ],
    }


def render_assistant(message: dict) -> None:
    if "error" in message:
        st.error(message["error"])
        return

    if message["rewritten"]:
        st.caption(f"חיפשתי: {message['search_question']}")
    st.markdown(message["content"])

    if not message["citations_ok"]:
        st.caption("לא ניתן היה לזהות את הקטעים שהמודל ציין; מוצגים כל המסמכים שנשלפו.")
    if message["sources"]:
        sources = "\n".join(f"- {title} ({doc_id})" for doc_id, title in message["sources"])
        st.markdown(f"**מקורות:**\n{sources}")
    else:
        st.caption("אין מקורות – המידע אינו מופיע במסמכים.")

    with st.expander(f"הקטעים שנשלפו ({len(message['chunks'])})"):
        for i, chunk in enumerate(message["chunks"], 1):
            st.markdown(f"**[קטע {i}] {chunk['title']} ({chunk['doc_id']})** · ציון RRF: {chunk['score']:.4f}")
            st.markdown(f"<div class='chunk'>{html.escape(chunk['text'])}</div>", unsafe_allow_html=True)


def render_sidebar() -> tuple:
    """מחזיר (שאלה לדוגמה אם נלחצה, האם נלחץ "שיחה חדשה")."""
    clicked = None
    with st.sidebar:
        new_chat = st.button("🔄 שיחה חדשה", width="stretch")
        st.header("על המערכת")
        st.markdown(
            "צ'אטבוט שעונה על שאלות על מוצרי ROBOGEN **על סמך 15 מסמכי החברה בלבד**. "
            "לכל שאלה נשלפים 6 קטעים רלוונטיים (חיפוש היברידי: וקטורי + BM25), "
            "והמודל עונה רק מתוכם ומציין את המקורות. אם המידע לא מופיע – המערכת אומרת זאת."
        )
        if config.CONVERSATION_MEMORY:
            st.markdown(
                f"**זיכרון שיחה:** שאלת המשך (למשל \"ומה לגבי SCOUT?\") מנוסחת מחדש לשאלה עצמאית "
                f"על סמך {config.MEMORY_TURNS} הסבבים האחרונים, והחיפוש נעשה עליה."
            )
        st.info("ROBOGEN היא חברה בדיונית, וכל התוכן במסמכים בדיוני ונוצר לצורך פרויקט אקדמי.")
        st.subheader("שאלות לדוגמה")
        for i, question in enumerate(SAMPLE_QUESTIONS):
            if st.button(question, key=f"sample_{i}", width="stretch"):
                clicked = question
    return clicked, new_chat


def main() -> None:
    st.set_page_config(page_title="ROBOGEN – עוזר מידע", page_icon="🤖")
    st.markdown(RTL_CSS, unsafe_allow_html=True)

    st.title("🤖 עוזר המידע של ROBOGEN")
    if not config.CONVERSATION_MEMORY:
        st.caption("כל שאלה נענית בנפרד – המערכת אינה זוכרת שאלות קודמות בשיחה.")

    sample, new_chat = render_sidebar()

    try:
        engine = load_engine()
    except Exception as error:
        logger.error("engine load failed: %s", type(error).__name__)
        st.error("לא ניתן לטעון את המערכת. ודאו שקובץ ‎.env‎ מכיל GEMINI_API_KEY "
                 "ושהאינדקס נבנה (python src/ingest.py).")
        st.stop()

    if "messages" not in st.session_state or new_chat:
        st.session_state.messages = []

    # ההיסטוריה על המסך; כשהזיכרון פעיל, הסבבים האחרונים משמשים לניסוח מחדש של שאלות המשך
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_assistant(message)

    question = st.chat_input("שאלו שאלה על מוצרי ROBOGEN...") or sample
    if not question:
        return

    history = conversation_history(st.session_state.messages)  # לפני הוספת השאלה הנוכחית
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("מחפש במסמכים..."):
            message = answer_question(engine, question, history)
        render_assistant(message)
    st.session_state.messages.append(message)


main()
