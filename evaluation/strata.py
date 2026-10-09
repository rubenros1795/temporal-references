"""
Strata of the evaluation, shared by 1_run_detector.py, 2_make_sample.py and 5_score.py.

Every test sentence of the cascade main run falls in exactly one parent stratum, by what the pipeline decided:

  A_<era>     counted as an analogy (split by the era of the crisis)
  B_split     the two verifiers disagree
  C_rejected  screen yes, both verifiers no
  D_neighbour both verifiers yes, but the quote lies outside the TARGET
  Z_original  screen no, but the original pipeline (A_original) found an in-sentence comparison
  E1_cue      screen no, the sentence has a past cue (eerder, nooit, sinds, 19xx, ...)
  E2_nocue    screen no, no cue

E1 and E2 are split once more by an independent detector (1_run_detector.py) into *_flag (detector yes) and
*_noflag. The detector runs on a random phase-1 sample of E1 and E2, and the size of each part is estimated
from it (two-phase sampling; 5_score.py carries that uncertainty).

Test sentences = the main run minus everything used while building the pipeline: earlier runs' samples,
excluded_ids_earlier_runs.txt and sentences that appear as few-shot examples.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config.py").exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "2_extraction" / "B_cascade"))
import config  # noqa: E402

RUN = config.CASCADE_OUT / "main"
DETECTOR_MANIFEST = RUN / "eval_detector_manifest.csv"
DETECTOR_JSONL = RUN / "eval_detector.jsonl"

# words that often signal a past in Dutch; only used to split the screen-no sentences
CUE = re.compile(r"\b(?:eerder|vroeger|destijds|indertijd|ooit|nooit|opnieuw|weer|wederom|sinds|sedert|geleden|"
                 r"herinner\w*|doet denken|deed denken|evenals|net als|zoals (?:in|bij|toen)|vorige|voorgaande|"
                 r"1[89]\d\d|toen)\b", re.I)
ERAS = ["pre1940", "1940_1969", "1970_1992"]


def era(year):
    return "pre1940" if year < 1940 else "1940_1969" if year < 1970 else "1970_1992"


def _norm(s):
    return " ".join(str(s).lower().split())


def dev_ids(d):
    """Sentences used while building the pipeline (never test sentences)."""
    base = config.CASCADE_OUT
    dev = set((base / "excluded_ids_earlier_runs.txt").read_text().split())
    for m in base.glob("*/sample_manifest.csv"):
        if m.parent != RUN:
            dev |= set(pd.read_csv(m, dtype=str).instance_id)
    from examples import EXAMPLES
    shots = [_norm(e["target"])[:60] for e in EXAMPLES]
    dev |= set(d.instance_id[d.target_sentence.map(lambda t: any(s in _norm(t) for s in shots))])
    return dev


def load_main():
    """The main run with parent stratum, era and dev flag per sentence."""
    d = pd.read_csv(RUN / "analogies.csv", dtype=str, keep_default_na=False)
    events = {e["label"]: e for e in json.load(open(config.EVENTS_JSON, encoding="utf-8"))}
    d["crisis"] = d.event.map(lambda e: events[e]["full name"])
    d["crisis_start"] = d.event.map(lambda e: events[e]["start_date"])
    d["crisis_year"] = d.crisis_start.str[-4:].astype(int)
    d["era"] = d.crisis_year.map(era)
    d["dev"] = d.instance_id.isin(dev_ids(d))
    d["cue"] = d.target_sentence.str.contains(CUE)
    orig = pd.read_csv(config.ORIGINAL_OUT / "comparisons_final.csv", dtype=str)
    orig_target = set(orig[orig.evidence_location == "target"].instance_id)

    yes = d.analogy == "True"
    screen_no = d.screen == "no"
    z = screen_no & d.instance_id.isin(orig_target)
    d["parent"] = ""
    d.loc[yes, "parent"] = "A_" + d.era[yes]
    d.loc[~yes & (d.verifiers == "split"), "parent"] = "B_split"
    d.loc[~yes & (d.verifiers == "both_no"), "parent"] = "C_rejected"
    d.loc[~yes & (d.verifiers == "both_yes"), "parent"] = "D_neighbour"
    d.loc[z, "parent"] = "Z_original"
    d.loc[screen_no & ~z & d.cue, "parent"] = "E1_cue"
    d.loc[screen_no & ~z & ~d.cue, "parent"] = "E2_nocue"
    missing = d.parent == ""
    if missing.any():
        sys.exit(f"{missing.sum()} sentences fall in no stratum (screen errors?); fix the main run first.")
    return d
