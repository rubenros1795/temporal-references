"""
Adjudication app (run locally by the adjudicator once every coder has finished).

    .venv_annotator/bin/python evaluation/pull_labels.py        # labels from GitHub, if the coding app ran there
    .venv_annotator/bin/streamlit run evaluation/adjudicate.py

Lists disagreements between two coders, items someone marked unsure and single-coded "yes" items. Answers are
shown without names, since the adjudicator may also be a coder. Decisions go to labels/_review.jsonl.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from labels_io import LABELS, append, label_file, read_labels  # noqa: E402
from app import CODEBOOK_VERSION, answer_form, load, show_item  # noqa: E402  (app.py only defines when imported)


def needs_decision(got, double):
    """Why an item goes to the adjudicator, or None."""
    if double:
        a = {r["analogy"] for r in got}
        if len(a) > 1:
            return "coders disagree"
        if a == {"yes"} and (len({r["direction"] for r in got}) > 1 or len({r["kind"] for r in got}) > 1):
            return "labels differ"
    elif got[0]["analogy"] == "yes":
        return "single-coded yes"
    if any(r.get("unsure") for r in got):
        return "unsure"
    return None


def main():
    items, asg, coders = load()
    adj = json.load(open(HERE / "data" / "setup.json", encoding="utf-8"))["adjudicator"]
    st.title("Adjudication")
    labs = {c: read_labels(label_file(c)) for c in coders}
    left = {c: sum(i not in labs[c] for i in asg[asg.coder == c].item_id) for c in coders}
    if any(left.values()):
        st.warning("Not every coder has finished (run pull_labels.py for the latest). Still to code: "
                   + ", ".join(f"{c} {n}" for c, n in left.items()))
        return
    rev_path = LABELS / "_review.jsonl"
    reviewed = read_labels(rev_path)
    todo = []
    for item_id, g in asg.groupby("item_id"):
        got = [labs[c][item_id] for c in g.coder]
        why = needs_decision(got, bool(g.double.iloc[0]))
        if why:
            todo.append((item_id, got, why))
    open_ = [t for t in todo if t[0] not in reviewed]
    st.progress(1 - len(open_) / max(len(todo), 1), text=f"{len(todo) - len(open_)} of {len(todo)} decided")
    lst = open_ if st.checkbox("only undecided", True) else todo
    if not lst:
        st.success("Everything decided. Run evaluation/score.py.")
        return
    pick = st.selectbox("item", range(len(lst)), format_func=lambda k: f"{lst[k][0]} ({lst[k][2]})")
    item_id, got, why = lst[pick]
    show_item(items.loc[item_id])
    shown = sorted(got, key=lambda r: hashlib.sha1((item_id + r["coder"]).encode()).hexdigest())
    st.table(pd.DataFrame([{"answer": f"coder {n}", **{k: r.get(k, "") for k in
                                                        ["analogy", "direction", "kind", "past", "unsure", "note"]}}
                           for n, r in enumerate(shown, 1)]).set_index("answer"))
    st.markdown("**Decision:**")
    rec = answer_form(f"r_{item_id}", reviewed.get(item_id, {}))
    if rec:
        append(rev_path, {"item_id": item_id, "coder": f"review:{adj}", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "codebook": CODEBOOK_VERSION, "reason": why, **rec})
        st.rerun()


main()
