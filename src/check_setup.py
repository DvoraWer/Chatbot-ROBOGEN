"""בדיקת הקמה: המפתח נטען, וקריאה בודדת ל-embedding ול-LLM עובדת."""
import config
from embedding import get_embed_model

print("1) מפתח API:", "נטען" if config.GEMINI_API_KEY else "חסר!")
config.require_api_key()

print(f"2) Embedding ({config.EMBED_MODEL})...")
embed_model = get_embed_model()
doc_vec = embed_model.get_text_embedding("ATLAS הוא רובוט מחסן.")
query_vec = embed_model.get_query_embedding("מה זה ATLAS?")
print(f"   ממד וקטור מסמך: {len(doc_vec)}, ממד וקטור שאלה: {len(query_vec)}")

print(f"3) LLM ({config.LLM_MODEL})...")
llm = config.get_llm()
print("   תשובה:", llm.complete("ענה במילה אחת בעברית: מה צבע השמיים ביום בהיר?").text.strip())

print("ההקמה תקינה.")
