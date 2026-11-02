"""השוואת לפני/אחרי בין שתי הרצות של run_eval (תצורה ראשית בלבד).

הרצה:
    python eval/compare_runs.py before after
    python eval/compare_runs.py holdout_before holdout_after
"""
import csv
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent


def load(tag: str):
    with open(EVAL_DIR / f"results_{tag}_retrieval.csv", encoding="utf-8-sig") as f:
        retrieval = {int(r["id"]): r for r in csv.DictReader(f) if r["config"] == "main"}
    answers = json.loads((EVAL_DIR / f"results_{tag}_answers.json").read_text(encoding="utf-8"))
    return retrieval, {a["id"]: a for a in answers}


def recall(row: dict) -> float:
    """חלק המסמכים הצפויים שנשלפו (מחושב מה-CSV, כך שעובד גם על הרצות ישנות)."""
    expected = set(row["expected_docs"].split())
    return len(expected & set(row["retrieved_docs"].split())) / len(expected)


def metrics(retrieval: dict):
    ranks = [int(r["first_rank"]) if r["first_rank"] else None for r in retrieval.values()]
    n = len(ranks)
    return (sum(1 for r in ranks if r) / n,
            sum(1 / r for r in ranks if r) / n,
            sum(recall(r) for r in retrieval.values()) / n)


def main() -> None:
    before_tag, after_tag = sys.argv[1], sys.argv[2]
    ret_b, ans_b = load(before_tag)
    ret_a, ans_a = load(after_tag)

    (hit_b, mrr_b, rec_b), (hit_a, mrr_a, rec_a) = metrics(ret_b), metrics(ret_a)
    print(f"שליפה (main): Hit@6 {hit_b:.2f} → {hit_a:.2f} | MRR {mrr_b:.3f} → {mrr_a:.3f}"
          f" | Recall@6 {rec_b:.3f} → {rec_a:.3f}\n")

    print(f"{'#':<4}{'type':<16}{'rank':>8}{'recall':>12}  {'sources before':<36}{'sources after'}")
    for qid, a in ans_a.items():
        if qid in ret_a:
            rank = f"{ret_b[qid]['first_rank'] or '-'}→{ret_a[qid]['first_rank'] or '-'}"
            rec = f"{recall(ret_b[qid]):.2f}→{recall(ret_a[qid]):.2f}"
        else:
            rank = rec = "n/a"
        src_b = " ".join(ans_b[qid]["sources"]) or "-"
        src_a = " ".join(a["sources"]) or "-"
        print(f"{qid:<4}{a['type']:<16}{rank:>8}{rec:>12}  {src_b:<36}{src_a}")


if __name__ == "__main__":
    main()
