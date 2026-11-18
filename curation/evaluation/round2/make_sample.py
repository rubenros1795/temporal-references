"""
Round 2 of the evaluation: one shared sample of 250 sentences, coded by all three coders under the new codebook.

    .venv_annotator/bin/python curation/evaluation/round2/make_sample.py

Drawn from the production cascade run (Qwen3.6-27B, data/cascade/main) and enriched with likely analogies, but as
a stratified random sample: every sentence has a known chance of selection (weight = stratum size / items drawn),
so the codes give corpus-level precision and rates for this run *and* for any later run of the cascade (e.g. with
the rewritten codebook and prompts). Recall is rough with only ~50 items outside the likely strata; round 1
(re-adjudicated) is the better source for recall.

Strata, in order of priority (each sentence in exactly one):
  A_<era>     the production run counts it as an analogy                       likely
  G_gemma     not counted now, but counted by the Gemma-4 12B run (round 1)    likely
  B_split     verifiers disagree, or both yes with the quote outside the TARGET likely
  C_rejected  screen yes, both verifiers no                                    other
  E1_cue      screen no, past cue in the TARGET (strata.CUE)                   other
  E2_nocue    screen no, no cue                                                other

Excluded: round-1 items (all 965), sentences used while building the cascade (strata.dev_ids: earlier runs,
excluded ids, few-shot examples).

Writes curation/evaluation/round2/data/: items.csv (what coders see), key.csv (hidden: stratum, weight, model
output), assignments.csv (every coder codes every item, in the same order), setup.json ("open": false: the app
shows a notice until it is set to true, once the codebook is final). Refuses to overwrite once
round2/labels/ holds codes.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL))
import strata  # noqa: E402
from strata import config  # noqa: E402

DATA = HERE / "data"
RUN = strata.ROOT / "data" / "cascade" / "main"
QUOTA = {"A_pre1940": 40, "A_1940_1969": 40, "A_1970_plus": 40, "G_gemma": 45, "B_split": 35,
         "C_rejected": 15, "E1_cue": 20, "E2_nocue": 15}
LIKELY = ("A_", "G_", "B_")


def era(year):
    return "pre1940" if year < 1940 else "1940_1969" if year < 1970 else "1970_plus"


def load():
    d = pd.read_csv(RUN / "analogies.csv", dtype=str, keep_default_na=False)
    events = {e["label"]: e for e in json.load(open(config.EVENTS_JSON, encoding="utf-8"))}
    missing = set(d.event) - set(events)
    if missing:
        sys.exit(f"events without metadata: {missing}")
    d["crisis"] = d.event.map(lambda e: events[e]["full name"])
    d["crisis_start"] = d.event.map(lambda e: events[e]["start_date"])
    d["crisis_year"] = d.crisis_start.str[-4:].astype(int)
    d["era"] = d.crisis_year.map(era)
    d["cue"] = d.target_sentence.str.contains(strata.CUE)
    gem = pd.read_csv(config.GEMMA_CASCADE_OUT / "main" / "analogies.csv", dtype=str, keep_default_na=False,
                      usecols=["instance_id", "analogy"])
    d["gemma_yes"] = d.instance_id.isin(gem.instance_id[gem.analogy == "True"])

    yes = d.analogy == "True"
    d["stratum"] = ""
    d.loc[yes, "stratum"] = "A_" + d.era[yes]
    rest = d.stratum == ""
    d.loc[rest & d.gemma_yes, "stratum"] = "G_gemma"
    rest = d.stratum == ""
    d.loc[rest & d.verifiers.isin(["split", "both_yes"]), "stratum"] = "B_split"
    rest = d.stratum == ""
    d.loc[rest & (d.verifiers == "both_no"), "stratum"] = "C_rejected"
    rest = d.stratum == ""
    d.loc[rest & (d.screen == "no") & d.cue, "stratum"] = "E1_cue"
    d.loc[rest & (d.screen == "no") & ~d.cue, "stratum"] = "E2_nocue"
    if (d.stratum == "").any():
        sys.exit(f"{(d.stratum == '').sum()} sentences fall in no stratum")

    old = set(pd.read_csv(EVAL / "data" / "key.csv", dtype=str).instance_id)
    d["excluded"] = d.instance_id.isin(old | strata.dev_ids(d))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coders", default="Ruben,Fien,Adriaan")
    ap.add_argument("--adjudicator", default="Ruben")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    coders = [c.strip() for c in args.coders.split(",")]
    labels = HERE / "labels"
    if labels.exists() and any(labels.iterdir()) and not args.force:
        sys.exit("round2/labels/ already holds codes: the sample is fixed.")

    d = load()
    t = d[~d.excluded].copy()
    print(f"{len(d):,} sentences in the run, {d.excluded.sum():,} excluded, {len(t):,} test sentences\n")
    picks = []
    for name, q in QUOTA.items():
        p = t[t.stratum == name]
        take = p.sample(min(q, len(p)), random_state=args.seed)
        picks.append(take.assign(stratum_size=len(p), coded=len(take), weight=len(p) / len(take)))
        print(f"{name:12s} {'likely' if name.startswith(LIKELY) else 'other ':6s} draw {len(take):3d} of "
              f"{len(p):8,d}  (weight {len(p) / len(take):8.1f})")
    s = pd.concat(picks).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    s["item_id"] = [f"R{i:03d}" for i in range(1, len(s) + 1)]
    s["likely"] = s.stratum.str.startswith(LIKELY)

    asg = pd.concat([pd.DataFrame({"item_id": s.item_id, "coder": c, "double": True,
                                   "order": range(1, len(s) + 1)})
                     for c in coders])  # same items, same (already shuffled) order for every coder

    DATA.mkdir(parents=True, exist_ok=True)
    s.rename(columns={"date": "article_date", "paper_title": "newspaper"})[
        ["item_id", "crisis", "crisis_start", "newspaper", "article_date", "days_after_event",
         "context_before", "target_sentence", "context_after"]].to_csv(DATA / "items.csv", index=False)
    s[["item_id", "instance_id", "stratum", "likely", "stratum_size", "coded", "weight", "event", "crisis_year",
       "era", "cue", "gemma_yes", "screen", "spotted", "verifiers", "verify_when", "check", "in_target", "analogy",
       "direction", "kind", "referent", "compared_with"]].to_csv(DATA / "key.csv", index=False)
    asg.sort_values(["coder", "order"]).to_csv(DATA / "assignments.csv", index=False)
    json.dump({"round": 2, "open": False, "coders": coders, "adjudicator": args.adjudicator, "seed": args.seed,
               "run": "data/cascade/main (Qwen3.6-27B)", "test_sentences": len(t),
               "test_sentences_by_era": t.era.value_counts().to_dict(),
               "stratum_sizes": t.stratum.value_counts().to_dict(), "quota": QUOTA},
              open(DATA / "setup.json", "w"), indent=2)
    print(f"\n{len(s)} items ({s.likely.sum()} from likely strata), each coded by {len(coders)} coders")
    print(s.groupby("era").size().to_string())
    print(s.groupby("crisis").size().sort_values(ascending=False).to_string())
    print(f"wrote {DATA}/")


if __name__ == "__main__":
    main()
