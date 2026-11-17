# Evaluating the analogy cascade

Hand-coding evaluation of `2_cascade` (main run): precision, recall and the true rate of analogies,
overall and per era. **`PROCEDURE.md` explains the design and every calculation**; this file is the quick start.

**Round 2 (from 2026-11-17):** the procedure after round 1 (meeting, one shared 250-item round, final evaluation) is in `round2/README.md`.

## Steps

All commands run from the project root with `.venv_annotator/bin/python` (`python` below).

| step | who | command |
|---|---|---|
| 1. detector (done 2026-10-09, Qwen3-14B, 30 min) | Ruben | `timeout 30m python curation/evaluation/1_run_detector.py --url http://127.0.0.1:8082 --zero-shot --model-name "Qwen3-14B (Q4_K_M)"` |
| 2. draw the sample (done; fixed once coding starts) | Ruben | `python curation/evaluation/2_make_sample.py --coders "Ruben,Fien,Adriaan" --adjudicator Ruben` |
| 3. coding | all three | the Streamlit app, personal link `…/?coder=<name>` |
| 4. fetch labels, adjudicate | Ruben | `python curation/evaluation/3_pull_labels.py`, then `.venv_annotator/bin/streamlit run curation/evaluation/4_adjudicate.py` |
| 5. score | Ruben | `python curation/evaluation/5_score.py` → `results/report.md`, `results/gold.csv` |
| 6. models | Ruben | `python curation/evaluation/6_models.py run --tag <model>` (with that model served), then `6_models.py score` → `results/models.md`: every local model through the cascade on the sample, weighted precision/recall, kappa per coder, model agreement |

After step 2, commit and push `data/items.csv`, `data/assignments.csv` and `data/setup.json` (never `key.csv`).

## The coding app

Deployed on Streamlit Community Cloud from `rubenros1795/temporal-references` (public), main file
`evaluation/app.py` (a two-line forwarder to `curation/evaluation/app.py`, kept so the deployment survived the move to `curation/`; redeploy with the real path to drop it), at `https://temporal-references-coding.streamlit.app`. Round 2: add `&round=2` to the coder link; it stays locked until `round2/data/setup.json` has `"open": true`; labels go to `labels/round2/<coder>.jsonl` on the `labels` branch. Only this folder is versioned (see
`../.gitignore`). The app's disk is wiped on every restart, so each answer is saved as a commit to
`labels/<coder>.jsonl` on the repo's `labels` branch (not `main`, which would redeploy the app). The app's secrets:

```
[github]
token = "github_pat_..."     # fine-grained, this repo only, Contents: read and write
repo = "rubenros1795/temporal-references"
branch = "labels"
```

Closing the browser loses at most the unsaved sentence on screen; the link resumes at the first uncoded one.
A failed save shows an error and does not move on. Locally without secrets, answers go to `labels/`.

## Files

```
curation/evaluation/
  README.md          this file
  PROCEDURE.md       design, calculations, relation to the cascade
  FOR_CODERS.md      one paragraph for the coders (shown in the app)
  CODEBOOK.md        coding rules (shown in the app; frozen once coding starts)
  app.py             coding app (deployed)
  labels_io.py       labels: local files or the GitHub labels branch
  strata.py          test sentences and strata
  1_run_detector.py  detector on the screen-no sentences -> data/cascade/main/eval_detector*
  2_make_sample.py   sample and assignments -> data/
  3_pull_labels.py   GitHub labels branch -> labels/
  4_adjudicate.py    adjudication app (local)
  5_score.py         -> results/report.md, results/gold.csv
  6_models.py        -> models/<tag>/, results/models.md (the sample was drawn from the Gemma-4 12B run; Qwen3.6-27B is production since 2026-11-01)
  data/
    items.csv        what coders see (crisis, article, context, TARGET)       [in git]
    assignments.csv  item -> coder, order, double                             [in git]
    setup.json       coders, adjudicator, quotas, detector, test counts       [in git]
    key.csv          stratum, weights, era, every model output                [local only]
  labels/            <coder>.jsonl (pulled), _review.jsonl (adjudication)     [local only]
  results/           report.md, gold.csv                                      [local only]
```
