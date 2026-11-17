"""
Run the cascade with other local models on the evaluation items, and score every model against the coders.

    # with the model's llama-server running on config.LLAMA_SERVER_URL:
    .venv_annotator/bin/python curation/evaluation/6_models.py run --tag qwen3-32b
    .venv_annotator/bin/python curation/evaluation/6_models.py score

`run` is the production cascade (2_cascade: screen -> verify + check -> quote in TARGET) with the
same prompts and few-shots, on all items in data/key.csv. Output: models/<tag>/{screen,verify,check}.jsonl and
decisions.csv. Resumable. Gemma-4 12B (production until 2026-10-31, the run the sample was drawn from) is not rerun: its answers are in key.csv.

`score` compares each model with each coder and with the gold of 5_score.py (disagreements as half a yes):
Cohen's kappa on the coded items, and precision/recall weighted to the corpus with the design weights. It also
scores the majority vote of the models, and the precision of the items every model calls an analogy.
Writes results/models.md.
"""
import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config.py").exists())
sys.path.insert(0, str(ROOT / "pipeline" / "2_cascade"))
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

MODELS = HERE / "models"
PROD = "gemma4-12b"  # the model the sample was drawn from


def run(tag, workers):
    import run as cascade  # pipeline/2_cascade/run.py
    key = pd.read_csv(HERE / "data" / "key.csv", dtype=str, keep_default_na=False)
    cols = ["instance_id", "event", "article_id", "date", "sentence_index", "target_sentence"]
    cand = pd.read_csv(config.CANDIDATES_CSV, dtype=str, usecols=cols).fillna("")
    sample = key[["item_id", "instance_id"]].merge(cand, on="instance_id")
    assert len(sample) == len(key), "some evaluation items are missing from the candidates file"
    cascade.OUT = MODELS / tag
    cascade.OUT.mkdir(parents=True, exist_ok=True)
    rows = sample.drop(columns=["item_id", "target_sentence"]).to_dict("records")

    screen = cascade.run_pass("screen", rows, workers)
    yes = [dict(r, spotted=screen[r["instance_id"]]["quote"]) for r in rows
           if screen.get(r["instance_id"], {}).get("compares") == "yes"]
    verify = cascade.run_pass("verify", yes, workers, spotted="spotted")
    check = cascade.run_pass("check", yes, workers)

    g = lambda src, f: sample.instance_id.map(lambda i: src.get(i, {}).get(f, ""))
    d = sample[["item_id", "instance_id", "target_sentence"]].assign(
        screen=g(screen, "compares"), spotted=g(screen, "quote"),
        verify_when=g(verify, "when"), check=g(check, "analogy"))
    d["in_target"] = [cascade.in_target(q, t) for q, t in zip(d.spotted, d.target_sentence)]
    d["analogy"] = (d.screen == "yes") & (d.verify_when == "before_crisis") & (d.check == "yes") & d.in_target
    missing = (d.screen == "") | ((d.screen == "yes") & ((d.verify_when == "") | (d.check == "")))
    d.drop(columns="target_sentence").to_csv(cascade.OUT / "decisions.csv", index=False)
    print(f"{tag}: {d.analogy.sum()} analogies among {len(d)} items; {missing.sum()} missing answers", file=sys.stderr)


def wpr(model, gold, w):
    """Horvitz-Thompson precision and recall (gold may be 0.5 for a provisional disagreement)."""
    tp = np.sum(w * model * gold)
    return tp / np.sum(w * model), tp / np.sum(w * gold)


def score():
    sc = importlib.import_module("5_score")
    key, asg, setup, labs, rev = sc.load()
    gold = sc.gold_table(asg, labs, rev)
    d = key.merge(gold, on="item_id")
    yn = lambda b: np.where(b, "yes", "no")

    preds = {PROD: d.analogy.astype(str).eq("True").to_numpy()}
    for f in sorted(MODELS.glob("*/decisions.csv")):
        m = pd.read_csv(f, dtype={"item_id": str}).set_index("item_id").analogy.astype(bool)
        if f.parent.name != PROD:
            preds[f.parent.name] = m.reindex(d.item_id).fillna(False).to_numpy()
    vote = np.mean(list(preds.values()), axis=0)
    others = [k for k in preds]
    preds[f"majority of {len(others)}"] = vote > 0.5
    preds[f"all {len(others)}"] = vote == 1
    preds[f"any of {len(others)}"] = vote > 0

    coders = setup["coders"]
    w = d.weight.astype(float).to_numpy()
    L = ["# Models on the evaluation items", "",
         f"{len(d)} fully coded items. Each model runs the production cascade (same prompts and few-shots); "
         f"{PROD} is the production run. Precision and recall are weighted to the corpus with the design weights; "
         "gold = adjudicated / agreed / single answer, a disagreement counts as half a yes (as in 5_score.py). "
         "Kappa is unweighted, on the coded items of that coder.", "",
         "| model | analogies (sample) | est. per 1,000 | precision | recall | F1 | "
         + " | ".join(f"kappa {c}" for c in coders) + " |",
         "|---|---|---|---|---|---|" + "---|" * len(coders)]
    n_test = w.sum()
    for name, m in preds.items():
        p_, r_ = wpr(m, d.pos.to_numpy(), w)
        ks = []
        for c in coders:
            col = f"coder:{c}"
            ok = d[col].notna() if col in d else pd.Series(False, index=d.index)
            ks.append(f"{sc.kappa(list(yn(m[ok])), list(d.loc[ok, col])):.2f} (n={ok.sum()})" if ok.sum() >= 20 else "–")
        L.append(f"| {name} | {m.sum()} | {np.sum(w * m) / n_test * 1000:.1f} | {p_:.2f} | {r_:.2f} | "
                 f"{2 * p_ * r_ / (p_ + r_):.2f} | " + " | ".join(ks) + " |")

    L += ["", "## Agreement between the models (kappa, all items)", "",
          "| | " + " | ".join(others) + " |", "|---|" + "---|" * len(others)]
    for a in others:
        L.append(f"| {a} | " + " | ".join(f"{sc.kappa(list(yn(preds[a])), list(yn(preds[b]))):.2f}" for b in others) + " |")

    L += ["", "## Is model agreement a quality signal?", "",
          "Among the items the production model counts: gold share by how many of the other models also count it.", "",
          "| other models agreeing | items | gold yes (unweighted) |", "|---|---|---|"]
    prod = preds[PROD]
    k_other = np.sum([preds[k] for k in others if k != PROD], axis=0)
    for k in range(len(others)):
        s = prod & (k_other == k)
        if s.any():
            L.append(f"| {k} of {len(others) - 1} | {s.sum()} | {d.pos[s].mean():.2f} |")
    out = HERE / "results" / "models.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True, help="folder name under models/, e.g. qwen3-32b")
    r.add_argument("--workers", type=int, default=8)
    sub.add_parser("score")
    a = ap.parse_args()
    run(a.tag, a.workers) if a.cmd == "run" else score()


if __name__ == "__main__":
    main()
