"""
Coding app for the cascade evaluation: one screen, one sentence at a time.

    .venv_annotator/bin/streamlit run evaluation/app.py

Each coder opens their own link (…/?coder=fien) and works through their own list. Every save is stored at once
(closing the browser loses at most the unsaved sentence on screen; the link resumes at the first uncoded one):
in labels/<coder>.jsonl locally, or, on Streamlit Cloud, on the `labels` branch of the GitHub repo set in the
app's secrets ([github] token, repo, branch); each save is a commit, so the history keeps every answer.
If a save fails, the app says so and does not move on. The latest answer per item counts; the codebook version is kept
with every answer. Coders never see model output, strata or each other's answers. Adjudication is a separate
app (adjudicate.py), run locally.
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
from labels_io import GitHubStore, LocalStore  # noqa: E402

DATA = HERE / "data"
DIRECTIONS = ["similarity", "rupture"]
KINDS = ["named_event", "series", "whole_past", "general_period"]
CODEBOOK = (HERE / "CODEBOOK.md").read_text(encoding="utf-8")
CODEBOOK_VERSION = hashlib.sha1(CODEBOOK.encode()).hexdigest()[:8]
INTRO = (HERE / "FOR_CODERS.md").read_text(encoding="utf-8").split("\n", 1)[1].strip()

st.set_page_config(page_title="Analogy coding", layout="centered")


@st.cache_data
def load():
    items = pd.read_csv(DATA / "items.csv", dtype=str, keep_default_na=False).set_index("item_id")
    asg = pd.read_csv(DATA / "assignments.csv", dtype={"item_id": str})
    coders = json.load(open(DATA / "setup.json", encoding="utf-8"))["coders"]
    return items, asg, coders


@st.cache_resource
def store():
    try:
        g = st.secrets["github"]
        return GitHubStore(g["token"], g["repo"], g.get("branch", "labels"))
    except (KeyError, FileNotFoundError):
        return LocalStore()


def show_item(it):
    st.caption(f"**{it.crisis}** (began {it.crisis_start}) · {it.newspaper} · {it.article_date}")
    if it.context_before:
        st.markdown(f"<div style='color:grey'>{it.context_before}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='border-left:4px solid #e0a800;padding:8px 12px;margin:6px 0;"
                f"background:rgba(224,168,0,.12)'><b>TARGET:</b> {it.target_sentence}</div>", unsafe_allow_html=True)
    if it.context_after:
        st.markdown(f"<div style='color:grey'>{it.context_after}</div>", unsafe_allow_html=True)


def answer_form(key, prev):
    """The coding form; returns the answer when submitted and complete, else None."""
    idx = lambda opts, v: opts.index(v) if v in opts else None
    with st.form(key):
        a = st.radio("Does the TARGET compare the present with a past from before the crisis?", ["yes", "no"],
                     index=idx(["yes", "no"], prev.get("analogy")), horizontal=True)
        c1, c2 = st.columns(2)
        dr = c1.radio("If yes: direction", DIRECTIONS, index=idx(DIRECTIONS, prev.get("direction")))
        kd = c2.radio("If yes: kind of past", KINDS, index=idx(KINDS, prev.get("kind")))
        past = st.text_input("If yes: the past referred to (a few words)", prev.get("past", ""))
        unsure = st.checkbox("unsure", prev.get("unsure", False))
        note = st.text_input("note (optional)", prev.get("note", ""))
        if not st.form_submit_button("Save and next", type="primary"):
            return None
    if a is None:
        st.error("Choose yes or no.")
        return None
    if a == "yes" and (dr is None or kd is None):
        st.error("With yes, also choose direction and kind of past.")
        return None
    yes = a == "yes"
    return {"analogy": a, "direction": dr if yes else "", "kind": kd if yes else "",
            "past": past.strip() if yes else "", "unsure": unsure, "note": note.strip()}


def main():
    items, asg, coders = load()
    by_key = {c.lower(): c for c in coders}
    coder = by_key.get(st.query_params.get("coder", "").lower())
    st.title("Analogies with the past")
    if coder is None:
        st.markdown(INTRO)
        coder = st.selectbox("Who are you?", coders, index=None)
        if coder is None:
            return
        st.query_params["coder"] = coder.lower()   # the link now remembers who you are
        st.rerun()

    mine = asg[asg.coder == coder].sort_values("order").item_id.tolist()
    if "done" not in st.session_state:
        try:
            st.session_state.done = store().read(coder)
        except Exception:
            st.error("Could not load your saved answers (connection problem). Nothing is lost: reload this page "
                     "in a minute.")
            return
    done = st.session_state.done
    if "pos" not in st.session_state:
        st.session_state.pos = next((k for k, i in enumerate(mine) if i not in done), len(mine))
    pos = st.session_state.pos

    n_done = sum(i in done for i in mine)
    if "saved" in st.session_state:
        st.toast(st.session_state.pop("saved"), icon="✅")
    st.progress(n_done / len(mine), text=f"{coder}: {n_done} of {len(mine)} coded")
    with st.expander("How this works"):
        st.markdown(INTRO)
    with st.expander("Codebook"):
        st.markdown(CODEBOOK)

    if pos >= len(mine):
        if n_done == len(mine):
            st.success("All done. Thank you! (You can still go back and change answers.)")
        else:
            st.session_state.pos = next(k for k, i in enumerate(mine) if i not in done)
            st.rerun()
        if st.button("← back"):
            st.session_state.pos = len(mine) - 1
            st.rerun()
        return

    item_id = mine[pos]
    show_item(items.loc[item_id])
    rec = answer_form(f"f_{item_id}", done.get(item_id, {}))
    if rec:
        rec = {"item_id": item_id, "coder": coder, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
               "codebook": CODEBOOK_VERSION, **rec}
        try:
            with st.spinner("saving…"):
                store().append(coder, rec)
        except Exception:
            # do not move on: the answer stays in the form, so clicking again retries
            st.error("⚠️ This answer was NOT saved (connection problem). Check your internet and click "
                     "'Save and next' again. Earlier answers are safe.")
            return
        done[item_id] = rec
        st.session_state.saved = f"Saved ({sum(i in done for i in mine)} of {len(mine)})"
        st.session_state.pos = next((k for k in range(pos + 1, len(mine)) if mine[k] not in done), len(mine))
        st.rerun()
    if pos > 0 and st.button("← back"):
        st.session_state.pos = pos - 1
        st.rerun()


if __name__ == "__main__":
    main()
