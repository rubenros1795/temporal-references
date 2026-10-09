"""
Coding app for the cascade evaluation: one screen, one sentence at a time.

    .venv_annotator/bin/streamlit run evaluation/app.py

Each coder opens their own link (…/?coder=fien) and works through their own list. Every save is stored at once
(closing the browser loses at most the unsaved sentence on screen; the link resumes at the first uncoded one):
in labels/<coder>.jsonl locally, or, on Streamlit Cloud, on the `labels` branch of the GitHub repo set in the
app's secrets ([github] token, repo, branch); each save is a commit, so the history keeps every answer.
If a save fails, the app says so and does not move on. The latest answer per item counts; the codebook version is kept
with every answer. Coders never see model output, strata or each other's answers. Adjudication is a separate
app (4_adjudicate.py), run locally.
"""
import hashlib
import html
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

st.set_page_config(page_title="Vergelijkingen met het verleden", layout="centered")
st.html("<style>section[data-testid='stSidebar']{width:430px !important}</style>")


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
    """The passage as one paragraph, with the TARGET sentence highlighted in place."""
    ctx = lambda t: f"<span style='color:#777'>{html.escape(t)}</span>" if t else ""
    target = (f"<mark style='background:#ffe08a;color:#111;padding:2px 3px;border-radius:3px'>"
              f"{html.escape(it.target_sentence)}</mark>")
    st.caption(f"**{it.crisis}** (begon {it.crisis_start}) · {it.newspaper} · {it.article_date}")
    st.html(f"<p style='font-size:1.12rem;line-height:1.75;margin:4px 0 16px'>"
            f"{ctx(it.context_before)} {target} {ctx(it.context_after)}</p>")


# stored values (English, used by the scoring) -> what coders see
NL = {"yes": "ja", "no": "nee", "similarity": "gelijkenis", "rupture": "breuk", "named_event": "één gebeurtenis",
      "series": "reeks", "whole_past": "hele verleden", "general_period": "periode"}


def answer_form(key, prev):
    """The coding form; returns the answer when submitted and complete, else None."""
    idx = lambda opts, v: opts.index(v) if v in opts else None
    nl = NL.get
    with st.form(key):
        a = st.radio("Vergelijkt de gemarkeerde zin het heden met een verleden van vóór de crisis?", ["yes", "no"],
                     index=idx(["yes", "no"], prev.get("analogy")), horizontal=True, format_func=nl)
        c1, c2 = st.columns(2)
        dr = c1.radio("Bij ja: richting", DIRECTIONS, index=idx(DIRECTIONS, prev.get("direction")), format_func=nl)
        kd = c2.radio("Bij ja: soort verleden", KINDS, index=idx(KINDS, prev.get("kind")), format_func=nl)
        past = st.text_input("Bij ja: welk verleden (een paar woorden)", prev.get("past", ""))
        unsure = st.checkbox("twijfel", prev.get("unsure", False))
        note = st.text_input("opmerking (optioneel)", prev.get("note", ""))
        if not st.form_submit_button("Opslaan en volgende", type="primary"):
            return None
    if a is None:
        st.error("Kies ja of nee.")
        return None
    if a == "yes" and (dr is None or kd is None):
        st.error("Kies bij ja ook de richting en het soort verleden.")
        return None
    yes = a == "yes"
    return {"analogy": a, "direction": dr if yes else "", "kind": kd if yes else "",
            "past": past.strip() if yes else "", "unsure": unsure, "note": note.strip()}


def main():
    items, asg, coders = load()
    by_key = {c.lower(): c for c in coders}
    coder = by_key.get(st.query_params.get("coder", "").lower())
    st.title("Vergelijkingen met het verleden")
    if coder is None:
        st.markdown(INTRO)
        coder = st.selectbox("Wie ben je?", coders, index=None)
        if coder is None:
            return
        st.query_params["coder"] = coder.lower()   # the link now remembers who you are
        st.rerun()

    mine = asg[asg.coder == coder].sort_values("order").item_id.tolist()
    if "done" not in st.session_state:
        try:
            st.session_state.done = store().read(coder)
        except Exception:
            st.error("Je opgeslagen antwoorden konden niet worden geladen (verbindingsprobleem). Er is niets "
                     "verloren: laad de pagina over een minuut opnieuw.")
            return
    done = st.session_state.done
    if "pos" not in st.session_state:
        st.session_state.pos = next((k for k, i in enumerate(mine) if i not in done), len(mine))
    pos = st.session_state.pos

    n_done = sum(i in done for i in mine)
    if "saved" in st.session_state:
        st.toast(st.session_state.pop("saved"), icon="✅")
    st.progress(n_done / len(mine), text=f"{coder}: {n_done} van {len(mine)} gedaan")
    with st.sidebar:
        st.markdown(CODEBOOK)
        with st.expander("Hoe werkt dit?"):
            st.markdown(INTRO)

    if pos >= len(mine):
        if n_done == len(mine):
            st.success("Klaar. Dank je wel! (Je kunt nog terug om antwoorden aan te passen.)")
        else:
            st.session_state.pos = next(k for k, i in enumerate(mine) if i not in done)
            st.rerun()
        if st.button("← vorige"):
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
            with st.spinner("opslaan…"):
                store().append(coder, rec)
        except Exception:
            # do not move on: the answer stays in the form, so clicking again retries
            st.error("⚠️ Dit antwoord is NIET opgeslagen (verbindingsprobleem). Controleer je internet en klik "
                     "nog eens op 'Opslaan en volgende'. Eerdere antwoorden zijn veilig.")
            return
        done[item_id] = rec
        st.session_state.saved = f"Opgeslagen ({sum(i in done for i in mine)} van {len(mine)})"
        st.session_state.pos = next((k for k in range(pos + 1, len(mine)) if mine[k] not in done), len(mine))
        st.rerun()
    if pos > 0 and st.button("← vorige"):
        st.session_state.pos = pos - 1
        st.rerun()


if __name__ == "__main__":
    main()
