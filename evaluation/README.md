# Evaluating the analogy cascade

How good is `2_extraction/B_cascade` (the main run)? We want three numbers, each with an interval:

- **precision**: of the analogies the pipeline counts, how many are real?
- **recall**: of all real analogies, how many does it count?
- **the true rate** of analogies per 1,000 sentences, overall and **per era**.

Plus diagnostics: which step loses the real analogies it misses, whether direction and kind labels are right,
and how well humans agree (the ceiling for any automatic method).

For coders, `FOR_CODERS.md` explains it all in one paragraph; the app shows it on its start page.

## Design

**Sample by what the pipeline decided.** Analogies are rare (2–3% of sentences), so random coding would find too
few. Every test sentence falls in one stratum by what the pipeline did. Each coded item is weighted by how many
sentences it stands for. Test sentences are the 139,344 sentences of the main run minus the 23,408 used while
building the pipeline (earlier runs' samples, excluded ids and few-shot sentences): 115,936.

| stratum | what the pipeline did | sentences | coded | coders |
|---|---|---|---|---|
| A (3 eras) | counted it as an analogy | 2,906 | 3 × 50 | 2 |
| B_split | the two verifiers disagreed | 1,708 | 70 | 2 |
| C_rejected | screen yes, both verifiers no | 2,985 | 50 | 2 |
| D_neighbour | both verifiers yes, quote outside the TARGET | 25 | all | 2 |
| Z_original | screen no, but the original pipeline found a comparison | 90 | 50 | 2 |
| E1_flag / E1_noflag | screen no, past cue; detector yes / no | 9,012 | 45 / 110 | 2 / 1 |
| E2_flag / E2_noflag | screen no, no cue; detector yes / no | 99,210 | 45 / 270 | 2 / 1 |

(The sizes above are from the main run of 2026-10-08. The E split depends on the detector; `make_sample.py` prints
the actual numbers.)

**Detector split of the screen-no sentences.** The screen-no strata hold over 90% of the sentences and dominate
the uncertainty of recall: in a stratum of 99k sentences, one real analogy among a few hundred coded items stands
for hundreds. `run_detector.py` asks the cascade's `check` question, zero-shot and with a different model
(Qwen3-14B), of a shuffled list of E1 and E2 sentences, stopped after 30 minutes. The sentences finished by then
are a random phase-1 sample. Sentences it flags form the `_flag` sub-strata and are sampled densely. Weights keep
every estimate unbiased whatever the detector's quality; the sub-stratum sizes are estimated from the phase-1
sample (two-phase sampling), and `score.py` carries that uncertainty.

**Who codes what (one list per coder, one round).** Each coder gets one list of about 417 sentences, in their own
random order, and is done when it is finished.
- Strata where judgement matters (A, B, C, D, Z and the flagged E strata; 435 items) are coded by **two** coders.
  The pairs rotate (Ruben–Fien, Ruben–Adriaan, Fien–Adriaan) so every pair overlaps on about 145 items. This gives
  agreement on all 435 items, not just a small shared set, and every gold label there rests on two people.
- The two big unflagged screen-no strata (380 items) are coded by **one**. They are almost all plain "no", and
  double coding them would halve the sample where it is needed most.
- **The adjudicator** (Ruben, who is also a coder) decides every item where two coders disagree, every item
  someone marked *unsure*, and every **single-coded yes**. In large strata a single false "yes" would count for
  hundreds of missed analogies, so none goes unchecked. The adjudicator works from the coders' answers, never
  from the model's: second-guessing only items where humans disagree with the model would bias the result
  towards the model. Answers are shown without names. Because Ruben also codes, his decisions could lean towards
  his own answers; the report's "Fien alone" and "Adriaan alone" estimates do not depend on adjudication and show
  whether that matters.
- Nobody sees model output or strata. Coders never see each other's answers (labels are stored in a private
  repo), and adjudication only starts when every coder has finished.

**Gold label per item:** the adjudicator's decision; otherwise the coders' agreed answer (double) or the single
coder's "no".

## Steps

1. **Detector** (once, 30 minutes; done on 2026-10-09 with the Qwen3-14B server on port 8082):
   ```
   timeout 30m .venv_annotator/bin/python evaluation/run_detector.py --url http://127.0.0.1:8082 --zero-shot \
       --model-name "Qwen3-14B (Q4_K_M)" > logs/eval_detector.log 2>&1
   ```
2. **Draw the sample** (done; refuses to redo it once coding has started):
   ```
   .venv_annotator/bin/python evaluation/make_sample.py --coders "Ruben,Fien,Adriaan" --adjudicator Ruben
   ```
3. **The coding app** runs on Streamlit Community Cloud from the private repo
   `Adapt-Preparing-societies-for-crises/temporal-references` (main file `evaluation/app.py`). Only this folder is
   versioned (see `../.gitignore`); `key.csv`, `labels/` and `results/` never leave this computer. The disk of a
   Streamlit Cloud app is wiped on every restart, so the app saves each answer as a commit to
   `labels/<coder>.jsonl` on the repo's `labels` branch (not `main`: a push to `main` redeploys the app). It needs
   a fine-grained GitHub token with *Contents: read and write* on this repo only, in the app's secrets:
   ```
   [github]
   token = "github_pat_..."
   repo = "Adapt-Preparing-societies-for-crises/temporal-references"
   branch = "labels"
   ```
   Each coder gets a personal link, `https://<app>.streamlit.app/?coder=fien`. Invite them as viewers of the app.
   Without secrets (locally: `.venv_annotator/bin/streamlit run evaluation/app.py`) answers go to `labels/`.
4. **Before they start:** walk the coders through `CODEBOOK.md` once (15 minutes), then **freeze it**. Every
   answer stores the codebook version; the report warns if more than one was used.
5. **Coding**, in their own time (about 4–5 hours each). They should not discuss sentences until all have finished.
6. **Adjudicate** (Ruben), locally, once all lists are done:
   ```
   .venv_annotator/bin/python evaluation/pull_labels.py      # labels branch -> labels/
   .venv_annotator/bin/streamlit run evaluation/adjudicate.py
   ```
   Decisions go to `labels/_review.jsonl`, which stays local.
7. **Score** (can be run at any point; partial results say so):
   ```
   .venv_annotator/bin/python evaluation/score.py
   ```
   Writes `results/report.md` and `results/gold.csv`.

## What the report gives

- **Agreement** on the 435 double-coded items: Krippendorff's alpha with a bootstrap interval, Cohen's kappa per
  pair, positive specific agreement, kappa weighted to the corpus, agreement on direction and kind. Also the
  **model as a coder**: its kappa with the humans next to the humans' kappa with each other.
- **Precision, recall, F1 and the true rate**, weighted to all test sentences, with Bayesian 95% intervals:
  each stratum's share of real analogies is drawn from its Beta posterior (Jeffreys prior). Unlike a bootstrap,
  a stratum with 0 positives still adds uncertainty, so recall is not overstated.
- **Per era:** precision, recall and the true rate. If recall differs by era, rates compared across eras need
  correcting, even if precision is flat.
- **Per stratum** and **real analogies lost per step** (verifiers, quote check, screen).
- **Robustness:** the estimates again with strict gold (yes only if everyone said yes), lenient gold (yes if
  anyone did) and each coder alone. Conclusions that hold in every row do not depend on who coded.
- **Labels:** direction and kind accuracy on correctly counted analogies, with a kind confusion table.

Known limit: in the single-coded strata a coder's false "no" is not caught. Its effect is at most the coder's miss
rate (seen on the double-coded items) times the few real analogies in those strata.

## Not covered here

- **Named events** (names, years, types) are checked in the gazetteer (`3_named_events/`, `gazetteer_checked.xlsx`).
- **Whether the quoted span is the right one.** We judge the sentence, not the quote.

## Files

```
evaluation/
  README.md         this file
  FOR_CODERS.md     one paragraph for the coders (shown in the app)
  CODEBOOK.md       coding rules (shown in the app)
  strata.py         stratum definitions, test-sentence selection
  run_detector.py   detector on the screen-no sentences -> output/2_extraction/B_cascade/main/eval_detector*.{csv,jsonl}
  make_sample.py    draws the sample -> data/
  app.py            Streamlit coding app (deployed)
  adjudicate.py     Streamlit adjudication app (local)
  pull_labels.py    labels from the GitHub `labels` branch -> labels/
  labels_io.py      reading/writing labels (local files or GitHub)
  score.py          -> results/report.md, results/gold.csv
  data/
    items.csv       what coders see (crisis, article, context, TARGET)
    key.csv         hidden: stratum, weights, era, every model output (local only)
    assignments.csv item -> coder, order, double
    setup.json      coders, adjudicator, seed, quotas, test-sentence counts
  labels/           one .jsonl per coder (pull_labels.py), _review.jsonl (adjudicate.py); local only
  results/          created by score.py
```
