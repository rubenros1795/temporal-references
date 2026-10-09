"""
Score the cascade against the hand codes.

    .venv_annotator/bin/python evaluation/5_score.py

Gold label per item: the adjudicator's decision if there is one; otherwise the agreed answer of the two coders
(double-coded items) or the single coder's answer. Until adjudication, a disagreement counts as half a yes and a
single-coded yes as a yes ("provisional" in the report).

Estimates are weighted to all test sentences of the main run. Intervals are Bayesian: per stratum the share of
real analogies is drawn from its Beta posterior (Jeffreys prior; unlike a bootstrap, a stratum with 0 positives
still contributes uncertainty), the split of those over eras from a Dirichlet, and for two-phase strata (E2)
the stratum size from the phase-1 detector counts. 20,000 draws.

Writes evaluation/results/report.md and gold.csv.
"""
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from labels_io import label_file, read_labels  # noqa: E402
from strata import ERAS  # noqa: E402

DATA, LABELS, OUT = HERE / "data", HERE / "labels", HERE / "results"
STAGE = {"A": "counted", "B": "verifiers disagree", "C": "both verifiers no", "D": "quote not in TARGET",
         "Z": "screen no", "E": "screen no"}
DRAWS = 20000


# ---------------------------------------------------------------- agreement statistics
def kappa(a, b, w=None):
    """Cohen's kappa, optionally with item weights (design weights -> corpus-level kappa)."""
    a, b = np.asarray(a), np.asarray(b)
    w = np.ones(len(a)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    po = w[a == b].sum()
    pe = sum(w[a == v].sum() * w[b == v].sum() for v in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def alpha(units):
    """Krippendorff's alpha, nominal. units: list of lists of values (>= 2 per unit)."""
    units = [u for u in units if len(u) >= 2]
    n_c = Counter(v for u in units for v in u)
    n = sum(n_c.values())
    if len(n_c) < 2:
        return float("nan")
    d_o = sum(sum(x != y for i, x in enumerate(u) for j, y in enumerate(u) if i != j) / (len(u) - 1) for u in units)
    d_e = sum(n_c[c] * n_c[k] for c in n_c for k in n_c if c != k) / (n - 1)
    return 1 - d_o / d_e


def boot_ci(stat, units, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = [stat([units[i] for i in rng.integers(0, len(units), len(units))]) for _ in range(reps)]
    return np.nanpercentile(v, [2.5, 97.5])


def psa(pairs):
    """Positive specific agreement over pairs of yes/no answers: 2a / (2a + b + c)."""
    a = sum(x == y == "yes" for x, y in pairs)
    bc = sum(x != y for x, y in pairs)
    return 2 * a / (2 * a + bc) if a + bc else float("nan")


# ---------------------------------------------------------------- gold
def load():
    key = pd.read_csv(DATA / "key.csv", dtype={"item_id": str}, keep_default_na=False)
    asg = pd.read_csv(DATA / "assignments.csv", dtype={"item_id": str})
    setup = json.load(open(DATA / "setup.json", encoding="utf-8"))
    labs = {c: read_labels(label_file(c)) for c in setup["coders"]}
    rev = read_labels(LABELS / "_review.jsonl")
    return key, asg, setup, labs, rev


def gold_table(asg, labs, rev):
    rows = []
    for item_id, g in asg.groupby("item_id"):
        got = [labs[c][item_id] for c in g.coder if item_id in labs[c]]
        if len(got) < len(g):
            continue  # not fully coded yet
        ans = [r["analogy"] for r in got]
        r_rev = rev.get(item_id)
        if r_rev:
            pos, src, lab = float(r_rev["analogy"] == "yes"), "adjudicated", r_rev
        elif len(set(ans)) == 1:
            pos, src = float(ans[0] == "yes"), "agreed" if len(got) > 1 else "single"
            lab = got[0]
            if len(got) == 1 and ans[0] == "yes":
                src = "single yes, provisional"
        else:
            pos, src, lab = 0.5, "disagreement, provisional", {}
        every = ans + ([r_rev["analogy"]] if r_rev else [])
        same = lambda f: lab.get(f, "") if len(got) == 1 or r_rev or len({r[f] for r in got}) == 1 else ""
        rows.append({"item_id": item_id, "n_coders": len(got), "source": src, "pos": pos,
                     "strict": float(all(a == "yes" for a in every)), "lenient": float(any(a == "yes" for a in every)),
                     "gold_direction": same("direction") if pos == 1 else "", "gold_kind": same("kind") if pos == 1 else "",
                     "gold_past": lab.get("past", "") if pos == 1 else "",
                     **{f"coder:{r['coder']}": r["analogy"] for r in got}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- estimation
def strata_info(key):
    return key.groupby("stratum").agg(size=("stratum_size", "first"), parent=("parent", "first"),
                                      parent_size=("parent_size", "first"), phase1_n=("phase1_n", "first"),
                                      phase1_sub=("phase1_sub", "first"), planned=("coded", "first"))


def point(m, info, col="pos"):
    """Plug-in estimates from coded items m (columns stratum, era, <col>)."""
    k = m.groupby("stratum")[col].sum()
    n = m.groupby("stratum").size()
    p = (k / n).reindex(info.index)
    if p.isna().any():
        return None  # some stratum has no coded items yet
    real = p * info["size"]
    is_a = info.index.str.startswith("A")
    tp, total = real[is_a].sum(), real.sum()
    return {"precision": tp / info["size"][is_a].sum(), "recall": tp / total if total else np.nan, "total": total}


def simulate(m, info, n_test, era_n, seed=0):
    """Posterior draws of precision, recall, rates and losses, overall and per era."""
    rng = np.random.default_rng(seed)
    sizes = {h: np.full(DRAWS, float(r["size"])) for h, r in info.iterrows()}
    two_phase = info[info.phase1_n < info.parent_size]   # size = parent size x share in the phase-1 sample
    for _, g in two_phase.groupby("parent"):
        share = rng.dirichlet(g.phase1_sub.to_numpy(float) + .5, DRAWS)
        for j, h in enumerate(g.index):
            sizes[h] = g.parent_size.iloc[0] * share[:, j]
    real = {e: np.zeros(DRAWS) for e in ERAS}       # real analogies per era, all strata
    tp = {e: np.zeros(DRAWS) for e in ERAS}
    model = {e: 0.0 for e in ERAS}
    lost = {}
    for h, r in info.iterrows():
        g = m[m.stratum == h]
        k, n = g.pos.sum(), len(g)
        p = rng.beta(k + .5, n - k + .5, DRAWS)
        # split the stratum's real analogies over eras: positives' eras, prior = eras of the coded items
        prior = np.array([(g.era == e).mean() for e in ERAS]) + 1e-9
        ke = np.array([g.pos[g.era == e].sum() for e in ERAS])
        share = rng.dirichlet(ke + prior, DRAWS)
        tot = sizes[h] * p
        for j, e in enumerate(ERAS):
            real[e] += tot * share[:, j]
            if h.startswith("A"):
                tp[e] += tot * share[:, j]
        if h.startswith("A"):
            model[h[2:]] += r["size"]
        else:
            lost[STAGE[h[0]]] = lost.get(STAGE[h[0]], 0) + tot
    out = {}
    TP, R = sum(tp.values()), sum(real.values())
    out["precision"] = TP / sum(model.values())
    out["recall"] = TP / R
    out["rate"] = R / n_test * 1000
    for e in ERAS:
        out[f"precision {e}"] = tp[e] / model[e] if model[e] else np.full(DRAWS, np.nan)
        out[f"recall {e}"] = tp[e] / real[e]
        out[f"rate {e}"] = real[e] / era_n.get(e, np.nan) * 1000
    out.update({f"lost: {k}": v for k, v in lost.items()})
    return out, sum(model.values()), model


def q(x):
    return np.nanpercentile(x, [2.5, 50, 97.5])


# ---------------------------------------------------------------- report
def main():
    key, asg, setup, labs, rev = load()
    coders = setup["coders"]
    gold = gold_table(asg, labs, rev)
    if gold.empty:
        sys.exit("no fully coded items yet")
    d = key.merge(gold, on="item_id")
    info = strata_info(key)
    L = ["# Evaluation of the analogy cascade (main run)", ""]
    p = L.append

    # --- progress and state
    prog = asg.assign(done=[i in labs[c] for i, c in zip(asg.item_id, asg.coder)]).groupby("coder").done.agg(["sum", "size"])
    p("## Progress\n")
    p(prog.rename(columns={"sum": "coded", "size": "assigned"}).to_markdown())
    complete = len(gold) == asg.item_id.nunique()
    prov = gold.source.str.contains("provisional").sum()
    p(f"\n{len(gold)} of {asg.item_id.nunique()} items fully coded; {len(rev)} adjudicated; {prov} still provisional.")
    if not complete or prov:
        p("\n*Partial results: not all items are coded and adjudicated yet.*")
    if setup["adjudicator"] in coders:
        others = [c for c in coders if c != setup["adjudicator"]]
        p(f"\nThe adjudicator ({setup['adjudicator']}) is also a coder. The estimates from "
          f"{' and '.join(others)} alone (Robustness) do not depend on adjudication.")
    if setup.get("detector"):
        p(f"\nE1/E2 split by detector: {setup['detector']['model']}, {setup['detector']['prompt']}.")
    versions = Counter(r.get("codebook", "?") for c in coders for r in labs[c].values())
    if len(versions) > 1:
        p(f"\n**Warning: answers were made under {len(versions)} codebook versions** ({dict(versions)}). "
          "The codebook should not change once coding has started.")
    p("")

    # --- agreement on double-coded items
    dbl = asg[asg.double].groupby("item_id").coder.apply(list)
    dbl = dbl[[all(i in labs[c] for c in cs) for i, cs in dbl.items()]]
    if len(dbl) >= 10:
        units = [[labs[c][i]["analogy"] for c in cs] for i, cs in dbl.items()]
        a_lo, a_hi = boot_ci(alpha, units)
        pairs = [tuple(u) for u in units]
        w = key.set_index("item_id").weight.reindex(dbl.index).to_numpy()
        p(f"## Agreement between coders ({len(dbl)} double-coded items)\n")
        p(f"- analogy yes/no: Krippendorff's alpha **{alpha(units):.2f}** (95% CI {a_lo:.2f}–{a_hi:.2f}); "
          f"raw agreement {np.mean([x == y for x, y in pairs]):.0%}; "
          f"positive specific agreement {psa(pairs):.2f} (how often a 'yes' by one coder is shared by the other)")
        p(f"- weighted to the corpus (design weights): kappa {kappa([x for x, _ in pairs], [y for _, y in pairs], w):.2f}. "
          "The unweighted figures describe this enriched sample, not the corpus.")
        p("\n| pair | items | Cohen's kappa | raw agreement |\n|---|---|---|---|")
        for x, y in combinations(coders, 2):
            ids = [i for i, cs in dbl.items() if set(cs) == {x, y}]
            if ids:
                a_, b_ = [labs[x][i]["analogy"] for i in ids], [labs[y][i]["analogy"] for i in ids]
                p(f"| {x}–{y} | {len(ids)} | {kappa(a_, b_):.2f} | {np.mean(np.array(a_) == np.array(b_)):.0%} |")
        p("")
        both = [(i, cs) for i, cs in dbl.items() if all(labs[c][i]["analogy"] == "yes" for c in cs)]
        if both:
            for f in ["direction", "kind"]:
                u = [[labs[c][i][f] for c in cs] for i, cs in both]
                p(f"- {f}, on the {len(both)} items both call an analogy: alpha {alpha(u):.2f}, "
                  f"same answer {np.mean([len(set(x)) == 1 for x in u]):.0%}")
        # the model as a further coder
        mod = key.set_index("item_id").analogy.astype(str).map({"True": "yes", "False": "no"})
        hum = [x for x, _ in pairs] + [y for _, y in pairs]
        mm = [mod[i] for i in dbl.index] * 2
        ww = np.concatenate([w, w])
        hh_k, hh_kw = kappa([x for x, _ in pairs], [y for _, y in pairs]), kappa([x for x, _ in pairs], [y for _, y in pairs], w)
        p(f"\n**The model as a coder** (same items): kappa model–human {kappa(mm, hum):.2f} vs human–human {hh_k:.2f}; "
          f"weighted to the corpus {kappa(mm, hum, ww):.2f} vs {hh_kw:.2f}. If the model is close to the humans, "
          "its errors are of the size of human disagreement.")
        p("")

    # --- main estimates
    m = d.copy()
    if any((m.stratum == h).sum() == 0 for h in info.index):
        p("Some strata have no coded items yet; no estimates.")
        write(L, d)
        return
    n_test, era_n = setup["test_sentences"], setup.get("test_sentences_by_era", {})
    sim, n_model, model_by_era = simulate(m, info, n_test, era_n)
    pt = point(m, info)
    p("## Pipeline quality, weighted to all test sentences\n")
    p(f"{n_test:,} test sentences (the main run minus development sentences); the pipeline counts {n_model:,.0f} "
      f"analogies among them ({n_model / n_test * 1000:.1f} per 1,000).\n")
    p("| measure | estimate | 95% interval |\n|---|---|---|")
    f1 = 2 * sim["precision"] * sim["recall"] / (sim["precision"] + sim["recall"])
    for name, x, est, fmt in [("precision (counted analogies that are real)", sim["precision"], pt["precision"], "{:.0%}"),
                              ("recall (real analogies that are counted)", sim["recall"], pt["recall"], "{:.0%}"),
                              ("F1", f1, 2 * pt["precision"] * pt["recall"] / (pt["precision"] + pt["recall"]), "{:.2f}"),
                              ("real analogies per 1,000 sentences", sim["rate"], pt["total"] / n_test * 1000, "{:.1f}")]:
        lo, _, hi = q(x)
        p(f"| {name} | {fmt.format(est)} | {fmt.format(lo)}–{fmt.format(hi)} |")
    p(f"| counted by the pipeline per 1,000 | {n_model / n_test * 1000:.1f} | |")
    p("")

    # --- per era
    p("## Per era of the crisis\n")
    p("If precision or recall differ between eras, the pipeline's rates are not comparable across eras as they stand; "
      "correct them by dividing the counted rate by recall/precision of the era.\n")
    p("| era | test sentences | counted per 1,000 | precision | recall | real per 1,000 |\n|---|---|---|---|---|---|")
    for e in ERAS:
        cell = lambda k, fmt: "{} ({}–{})".format(*[fmt.format(v) for v in (q(sim[k])[1], q(sim[k])[0], q(sim[k])[2])])
        n_e = era_n.get(e, 0)
        p(f"| {e} | {n_e:,} | {model_by_era[e] / n_e * 1000 if n_e else float('nan'):.1f} | "
          f"{cell(f'precision {e}', '{:.0%}')} | {cell(f'recall {e}', '{:.0%}')} | {cell(f'rate {e}', '{:.1f}')} |")
    p("\n(median and 95% interval of the posterior)\n")

    # --- per stratum and where real analogies are lost
    rng = np.random.default_rng(1)
    st_ = m.groupby("stratum").agg(coded=("pos", "size"), real=("pos", "sum"))
    st_ = info.join(st_)
    st_["share_real"] = st_.real / st_.coded
    st_["95% interval"] = [("{:.1%}–{:.1%}".format(*np.percentile(rng.beta(k + .5, n - k + .5, DRAWS), [2.5, 97.5])))
                           for k, n in zip(st_.real, st_.coded)]
    st_["est_real"] = (st_.share_real * st_["size"]).round(0)
    st_["stage"] = st_.index.str[0].map(STAGE)
    p("## Per stratum: share of real analogies\n")
    p(st_[["stage", "size", "coded", "real", "share_real", "95% interval", "est_real"]]
      .rename(columns={"size": "sentences"}).round({"sentences": 0, "share_real": 3}).to_markdown())
    p("\nReal analogies the pipeline loses, by stage (posterior median and 95% interval):\n")
    for k in [k for k in sim if k.startswith("lost: ")]:
        lo, med, hi = q(sim[k])
        p(f"- {k[6:]}: {med:,.0f} ({lo:,.0f}–{hi:,.0f})")
    p("")

    # --- sensitivity to the gold standard
    p("## Robustness: other ways of deciding the gold label\n")
    p("Point estimates. If the conclusions hold in every row, they do not depend on how disagreements are settled "
      "or on any single coder.\n")
    p("| gold | items | precision | recall | real per 1,000 |\n|---|---|---|---|---|")
    variants = [("adjudicated (main)", m, "pos"), ("strict: yes only if everyone said yes", m, "strict"),
                ("lenient: yes if anyone said yes", m, "lenient")]
    for c in coders:
        col = f"coder:{c}"
        if col in m:
            mc = m[m[col].isin(["yes", "no"])].assign(**{"own": lambda x: (x[col] == "yes").astype(float)})
            variants.append((f"{c} alone", mc, "own"))
    for name, mm_, col in variants:
        r = point(mm_, info, col)
        if r is None:
            p(f"| {name} | {len(mm_)} | (a stratum has no items) | | |")
        else:
            p(f"| {name} | {len(mm_)} | {r['precision']:.0%} | {r['recall']:.0%} | {r['total'] / n_test * 1000:.1f} |")
    p("\nIn the single-coded strata, a coder's false 'no' is not caught. That bias is at most the coder's miss rate "
      "times the real analogies there, which is small because those strata are almost empty. Every single-coded "
      "'yes' is checked by the adjudicator.")
    p("")

    # --- labels on correctly counted analogies
    a = m[m.stratum.str.startswith("A") & (m.pos == 1) & (m.gold_kind != "")].copy()
    if len(a):
        p(f"## Labels on correctly counted analogies ({len(a)} items, weighted by era)\n")
        for f in ["direction", "kind"]:
            ok = (a[f] == a[f"gold_{f}"])
            p(f"- {f}: model agrees with gold on {np.average(ok, weights=a.weight):.0%}")
        p("\nKind, model (rows) by gold (columns):\n")
        p(pd.crosstab(a.kind, a.gold_kind).to_markdown())
        p("")

    p("## Gold labels used\n")
    p(d.source.value_counts().rename("items").to_frame().to_markdown())
    write(L, d)


def write(L, d):
    OUT.mkdir(exist_ok=True)
    d.to_csv(OUT / "gold.csv", index=False)
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {OUT / 'report.md'} and gold.csv")


if __name__ == "__main__":
    main()
