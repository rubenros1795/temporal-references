# Evaluation round 2 (and how we got here)

This file records the evaluation procedure from the end of round 1 (2026-11-17) onwards: what we found, what we
decided and why, and every command needed to reproduce it. Round 1's design and calculations are in
`../PROCEDURE.md`; the quick start for round 1 is `../README.md`.

## The procedure in short

1. **Round 1** (done): 965 items, stratified on the Gemma-4 12B run, coded by Ruben, Fien and Adriaan (435
   double-coded, 530 single-coded). Becomes **development material**.
2. **Meeting 2026-11-18**: decide the codebook rules on the grey zones found in round 1 (below). These decisions
   are **final**.
3. **Codebook rewrite**: put the decisions into `CODEBOOK.md` with examples. Wording and examples only, **no new
   rules** after round 2 coding has started. Examples come from outside the 250 round-2 items.
4. **Round 2**: 250 items, **all three coders code all 250**, blind (no model output, no stratum). This is the
   last coding round for the colleagues.
5. **Adjudication**: Ruben decides disagreements and *twijfel* items in round 2 under the final codebook, blind
   to names and model output. Ruben also **re-checks round 1** under the final codebook (alone).
6. **Prompts**: the cascade prompts get the same rules and examples (examples from outside the 250; tuning only
   on round 1, never on round 2). Run the new cascade on all candidates.
7. **Final scores**:
   - from round 2: agreement among the three coders, coder–model agreement, **precision per era** and the
     counted/true rate, weighted to the corpus;
   - from round 1 (re-adjudicated): **recall**. Round 2 has too few items outside the likely strata to measure it.
   - Recall is low and is **reported, not solved**. In the paper: absolute levels are uncertain; correctly counted
     analogies (count × precision) are a conservative minimum (the raw count is not, since ~4 in 10 are false);
     era comparisons are made after correcting each era by its precision (and recall); the missed analogies lean
     to series and similarity (implicit precedents without a comparison word), so the mix of kinds leans towards
     explicit forms.

## Why this design

- **Two rounds only.** Coder time is the constraint. Round 2 therefore does two jobs: agreement under the new
  rules *and* final evaluation.
- **Freeze the rules before coding.** If the codebook changed after round 2 was coded, the codes and the reported
  codebook would no longer match. If a rule must change anyway, round 2 is adjudicated under the final codebook
  and the agreement score is reported as agreement under the meeting's rules.
- **Enriched but random.** About 200 of 250 items come from strata where an analogy is likely, so the agreement
  score is about the "yes" cases that matter (round 1 was mostly easy "no"s). The items are drawn **at random
  within strata with known sizes**, not hand-picked: hand-picked items would only contain easy-to-spot analogies
  (circular for the model) and cannot give corpus-level precision or rates. Report round-2 agreement as
  "agreement on an enriched sample".
- **Valid for later model runs.** A stratified random sample with known weights gives unbiased estimates for any
  classifier, not only the one the strata were built from. So the cascade can be re-prompted and re-run after
  round 2 and still be scored on it, as long as round 2 items are never used for tuning.
- **Clean test set.** Round-1 items and every sentence used while building the cascade are excluded.

## Round 1: findings that led here (2026-11-17)

Full analysis with every item: `../results/round1_error_analysis/error_analysis.html` (also published at
https://claude.ai/artifact/6Dbrsnv4AKq21D9x2GV7Ca, private). Numbers are provisional (no adjudication).

- Agreement on 435 double-coded items: Krippendorff's alpha 0.46 (0.36–0.54), raw 79%, positive specific
  agreement 0.60. Kappa per pair: Ruben–Adriaan 0.65, Fien–Adriaan 0.39, Ruben–Fien 0.32.
- Gemma-4 12B run: precision 50% (40–59), recall 38% (26–49), 32.7 real analogies per 1,000.
  Qwen3.6-27B (production since 2026-11-01): precision 0.59, recall 0.43, F1 0.50 (`../results/models.md`).
- Precision per era (Gemma): pre-1940 41%, 1940–69 55%, 1970–92 49%. Recall per era similar (37/36/39%).
- Recall depends on 8 single-coded "yes" answers; 5 in E2_noflag weigh 229 each. Three look like slips (S0004,
  S0257, S0700): reversing them moves recall 38% → ~46% and the rate 32.7 → 26.8.
- 93 disagreements, sorted by hand (`cats.json`): ~38 are slips against the codebook as written (superlatives,
  explicit comparisons, neighbouring sentence, implicit comparison); ~55 are grey zones.
- Production model, clear errors: 33 false positives (repetition words, past as background, statistics and side
  topics, before/after states), 25 false negatives (mainly named precedents without a comparison word; three
  doubtful single-coded items).

### The rules to decide on 2026-11-18

1. **Named or compared?** A past event beside the present without "zoals"/"net als" ("gezien de ervaringen op
   Walcheren"). Most common disagreement (19) and most common model miss (12).
2. **Must the present be the crisis?** "Omzetten hoger dan in februari 1952" in a flood article.
3. **"Vroeger" as a description** of how something used to work.
4. **"Weer"/"opnieuw"/"andermaal"**: the codebook says yes if the earlier case is before the crisis; coders
   mostly said no.
5. **Highlighted-sentence rule**: does "ook nu weer", pointing back to the previous sentence, count?

Record the decisions here, under "Decisions", before rewriting `CODEBOOK.md`.

## Decisions (2026-11-18)

Settled by Ruben on 2026-11-18 (instead of in the meeting). Put into `CODEBOOK.md` the same day; the examples there
are round-1 items, so none of the 250 round-2 items.

1. **Named or compared?** A comparison word is not needed: an earlier event placed beside the crisis counts (as
   parallel, lesson, precedent, yardstick or contrast). But it must be set beside the present: a past that is only
   background (biography, cause, date) stays *nee* (S0212, Mulder's Wijster legend). S0524, S0925, S0291 → ja.
2. **Present = the crisis?** There must be a demonstrable link between the present in the sentence and the crisis
   (the crisis or an element of it: course, consequences, damage, aid, reactions, people involved). Comparisons on
   other topics in the same article are *nee* (S0463, S0757). This also filters the few articles that are not about
   the crisis at all.
3. **"Vroeger".** Any past is fine, including an unspecified "vroeger", but it must be compared with the present
   crisis or an element of it (S0388, S0521 → ja).
4. **"Weer"/"opnieuw"/"andermaal".** The word alone is not enough: the earlier case must be clear from the sentence
   or its context. Coders are not asked to supply background knowledge (S0167 → nee; S0870 → ja).
5. **Highlighted-sentence rule.** The comparison should sit in the TARGET, but leniently: if the TARGET takes part in a
   comparison completed by the surrounding sentences, it counts (S0001 → ja).
6. **Kind of past.** Keep the four kinds. "Grootste ramp ooit" = *hele verleden*; "vroeger" is *periode* only when a
   bounded period is meant, otherwise *hele verleden*.

## Round 2 sample (drawn 2026-11-17)

    .venv_annotator/bin/python curation/evaluation/round2/make_sample.py          # seed 2026, reproducible

Population: the production cascade run `data/cascade/main` (Qwen3.6-27B), 261,646 sentences; 24,373 excluded
(round-1 items and `strata.dev_ids`), **237,273 test sentences**. Strata in order of priority, each sentence in
exactly one:

| stratum | definition | likely? | size | drawn | weight |
|---|---|---|---|---|---|
| A_pre1940 | Qwen counts it, crisis before 1940 | likely | 1,768 | 40 | 44.2 |
| A_1940_1969 | Qwen counts it, 1940–69 | likely | 2,017 | 40 | 50.4 |
| A_1970_plus | Qwen counts it, 1970 and later (incl. 1995 floods) | likely | 2,211 | 40 | 55.3 |
| G_gemma | not counted by Qwen, counted by the Gemma-4 12B run | likely | 2,410 | 45 | 53.6 |
| B_split | verifiers disagree, or both yes with quote outside TARGET | likely | 2,169 | 35 | 62.0 |
| C_rejected | screen yes, both verifiers no | other | 6,029 | 15 | 401.9 |
| E1_cue | screen no, past cue in TARGET (`strata.CUE`) | other | 19,313 | 20 | 965.6 |
| E2_nocue | screen no, no cue | other | 201,356 | 15 | 13,423.7 |

Expect roughly 90 real analogies among the 200 likely items (by round-1 precision). All 31 crises appear,
including four not in round 1 (Praag, R101, rivieren1995, Los Alfaques); eras 77 / 89 / 84 items.

Files in `data/`:

- `items.csv`: what coders see (crisis, newspaper, date, context, TARGET); ids R001–R250;
- `key.csv`: stratum, size, weight and Qwen's output per item (**local only**, in `.gitignore`);
- `assignments.csv`: every coder × every item, same order for all coders (the shuffled R001–R250 order);
- `setup.json`: coders, adjudicator, seed, run, stratum sizes, quota.

Do not redraw once coding has started (the script refuses when `round2/labels/` holds codes).

## Still to do

- [x] Record the decisions above; rewrite `CODEBOOK.md` (2026-11-18; new version hash in every answer).
- [x] `app.py`: round switch (`&round=2`, reads `round2/data/`, own label files on the `labels` branch).
- [x] Open round 2 (2026-11-18): `"open": true`, pushed. Send the links.
- [ ] Pull and adjudicate round 2 (adapt `3_pull_labels.py`, `4_adjudicate.py` to the round folder).
- [ ] Ruben re-checks round 1 under the final codebook (`../labels/_review.jsonl`), then `5_score.py` for recall.
- [ ] Rewrite the cascade prompts (examples outside the 250), run on all candidates.
- [ ] Scoring script for round 2: three-coder alpha, coder–model kappa, weighted precision and rates per era,
      by stratum weights from `key.csv` (adapt `5_score.py`; single-phase strata, no detector).

## Reproducing the round-1 error analysis

Everything is in `../results/round1_error_analysis/` (local only):

- `build_err.py`: joins items, gold, labels and Qwen3.6-27B's answers into `err_items.json`.
  Run from `curation/evaluation/`: `../../.venv_annotator/bin/python results/round1_error_analysis/build_err.py results/round1_error_analysis`
- `cats.json`: hand categories of the 93 disagreements (M mentioned, V before/after, N neighbour, U unprecedented,
  I implicit, W weer/opnieuw, C clear slip, S present not the crisis, T timing, O OCR). The false-positive and
  false-negative categories are in `gen.py` (`FPC`, `FNC`).
- `show.py dis|fp|fn 0 40`: prints items for reading.
- `gen.py` + `tpl.html` → `error_analysis.html` (run in that folder).
- `labels_local_before_pull_2026-11-17/`: the local labels before the final pull from GitHub (Adriaan had 39
  then); `report_before_pull_2026-10-31.md`: the report from those labels.
- Labels were pulled 2026-11-17 with `3_pull_labels.py` (Ruben 491 lines, of which 24 test answers from
  2026-10-09 on items outside his list, ignored by scoring; Fien 467; Adriaan 466).
