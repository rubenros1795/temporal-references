"""
Draw the evaluation sample from the cascade main run and give each coder one list.

    timeout 30m .venv_annotator/bin/python evaluation/1_run_detector.py --url ... --zero-shot   # once, first
    .venv_annotator/bin/python evaluation/2_make_sample.py --coders "Ruben,Fien,Adriaan" --adjudicator Ruben

Writes evaluation/data/:
  items.csv        what coders see: crisis, article, context, TARGET sentence (no model output)
  key.csv          hidden: stratum, weights, era and every model output per item
  assignments.csv  item -> coder, order, double (coded by two coders)
  setup.json       coders, adjudicator, seed, quotas

Strata (strata.py, README.md). Strata where judgement matters are coded by two coders (rotating over the
three pairs, so every pair overlaps about equally); the two big screen-no strata without a detector flag,
almost all plain "no", are coded by one. Every disagreement, every "unsure" and every single-coded "yes" goes
to the adjudicator, who sees the answers without names. Coders code one list, once.

Re-running with the same --seed gives the same sample; it refuses to overwrite once coding has started
(labels/ not empty) unless --force.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import strata  # noqa: E402

DATA = HERE / "data"

# stratum -> (items to code, coders per item); None = all
QUOTA = {"A_pre1940": (50, 2), "A_1940_1969": (50, 2), "A_1970_1992": (50, 2),
         "B_split": (70, 2), "C_rejected": (50, 2), "D_neighbour": (None, 2), "Z_original": (50, 2),
         "E1_flag": (45, 2), "E2_flag": (45, 2), "E1_noflag": (110, 1), "E2_noflag": (270, 1)}
# without the detector (--no-detector): E1/E2 unsplit, single-coded
QUOTA_NODET = {**{k: v for k, v in QUOTA.items() if not k.startswith("E")},
               "E1_cue": (150, 1), "E2_nocue": (270, 1)}


def detector_split(t):
    """Sub-stratum and phase-1 counts for the screen-no sentences, from 1_run_detector.py's output."""
    if not strata.DETECTOR_JSONL.exists():
        sys.exit("No detector output: run evaluation/1_run_detector.py first (or pass --no-detector).")
    man = pd.read_csv(strata.DETECTOR_MANIFEST, dtype=str)
    det = {r["instance_id"]: r for r in map(json.loads, open(strata.DETECTOR_JSONL, encoding="utf-8"))
           if not r.get("annotation_error")}
    # the list is shuffled, so the sentences finished so far are a random sample of it: phase 1 = those
    man = man[man.instance_id.isin(det)].copy()
    print(f"detector: {len(man):,} phase-1 sentences ({man.parent.value_counts().to_dict()})")
    man["flag"] = man.instance_id.map(lambda i: det[i]["analogy"] == "yes")
    man["stratum"] = man.parent.str[:2] + man.flag.map({True: "_flag", False: "_noflag"})
    return man.set_index("instance_id").stratum, man.groupby("parent").size(), man.groupby("stratum").size()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coders", required=True, help="three names, comma-separated")
    ap.add_argument("--adjudicator", required=True, help="decides disagreements (best not one of the coders)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--no-detector", action="store_true", help="do not split E1/E2 (not recommended)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    coders = [c.strip() for c in args.coders.split(",") if c.strip()]
    if len(coders) < 2:
        sys.exit("Give at least two coders.")
    if args.adjudicator in coders:
        print(f"Note: {args.adjudicator} codes and adjudicates. The adjudication app hides who gave which answer, "
              "and 5_score.py reports estimates from each other coder alone, which adjudication cannot affect.")
    labels = HERE / "labels"
    if labels.exists() and any(labels.iterdir()) and not args.force:
        sys.exit("labels/ already holds codes: the sample is fixed. Use --force only if you mean to start over.")

    d = strata.load_main()
    t = d[~d.dev].copy()
    t["stratum"] = t.parent
    t["parent_size"] = t.groupby("parent").instance_id.transform("size")
    t["phase1_n"] = t.parent_size        # phase-1 sentences in the parent stratum (= all, except E2)
    t["phase1_sub"] = 0                  # phase-1 sentences in the sub-stratum
    quota = QUOTA_NODET if args.no_detector else QUOTA
    if not args.no_detector:
        sub, n1_parent, n1_sub = detector_split(t)
        e = t.parent.str.startswith("E")
        t.loc[e, "stratum"] = t.instance_id[e].map(sub).fillna("")   # E2 outside phase 1 -> ""
        t = t[t.stratum != ""].copy()
        e = t.parent.str.startswith("E")
        t.loc[e, "phase1_n"] = t.parent[e].map(n1_parent)
        t.loc[e, "phase1_sub"] = t.stratum[e].map(n1_sub)

    picks = []
    for name, (q, k) in quota.items():
        p = t[t.stratum == name]
        n = len(p) if q is None else min(q, len(p))
        take = p.sample(n, random_state=args.seed)
        if name.startswith("E") and not args.no_detector:
            parent_size = p.parent_size.iloc[0]
            size = parent_size * p.phase1_sub.iloc[0] / p.phase1_n.iloc[0]   # estimated sub-stratum size
        else:
            size = len(p)
        picks.append(take.assign(stratum_size=size, coded=n, weight=size / max(n, 1), n_coders=k))
        print(f"{name:12s} code {n:4d} x{k} of {size:9,.0f}  (weight {size / max(n, 1):7.1f})")
    s = pd.concat(picks).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    s["item_id"] = [f"S{i:04d}" for i in range(1, len(s) + 1)]

    # assignments: double-coded items rotate over the coder pairs within each stratum; single ones round-robin
    pairs = [(a, b) for i, a in enumerate(coders) for b in coders[i + 1:]]
    rows, r2, r1 = [], 0, 0
    for _, g in s.groupby("stratum", sort=True):
        for item_id, k in zip(g.item_id, g.n_coders):
            if k == 2:
                who, r2 = pairs[r2 % len(pairs)], r2 + 1
            else:
                who, r1 = (coders[r1 % len(coders)],), r1 + 1
            rows += [{"item_id": item_id, "coder": c, "double": k == 2} for c in who]
    asg = pd.DataFrame(rows)
    asg["order"] = 0
    for k, c in enumerate(coders):
        mine = asg.coder == c
        asg.loc[mine, "order"] = pd.Series(range(1, mine.sum() + 1)).sample(frac=1, random_state=args.seed + k).values

    DATA.mkdir(parents=True, exist_ok=True)
    s.rename(columns={"date": "article_date", "paper_title": "newspaper"})[
        ["item_id", "crisis", "crisis_start", "newspaper", "article_date", "days_after_event",
         "context_before", "target_sentence", "context_after"]].to_csv(DATA / "items.csv", index=False)
    s[["item_id", "instance_id", "stratum", "parent", "stratum_size", "parent_size", "phase1_n", "phase1_sub",
       "coded", "weight", "n_coders", "event", "crisis_year", "era", "cue", "screen", "spotted", "verify_when",
       "check", "verifiers", "in_target", "analogy", "direction", "kind", "referent"]].to_csv(DATA / "key.csv", index=False)
    asg.sort_values(["coder", "order"]).to_csv(DATA / "assignments.csv", index=False)
    json.dump({"coders": coders, "adjudicator": args.adjudicator, "seed": args.seed, "detector": None if args.no_detector else json.load(open(strata.RUN / "eval_detector_info.json")),
               "test_sentences": int((~d.dev).sum()),
               "test_sentences_by_era": d[~d.dev].era.value_counts().to_dict(), "quota": {k: list(v) for k, v in quota.items()}},
              open(DATA / "setup.json", "w"), indent=2)
    per = asg.groupby("coder").agg(items=("item_id", "size"), double=("double", "sum"))
    print(f"\n{len(s)} items ({(s.n_coders == 2).sum()} double-coded), {len(asg)} codings")
    print(per.to_string())
    print(f"wrote {DATA}/items.csv, key.csv, assignments.csv, setup.json")


if __name__ == "__main__":
    main()
