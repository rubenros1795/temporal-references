# Evaluation procedure

What we measure, how the sample is drawn, how each number is calculated, and how all of it maps onto the
cascade. Scripts: `strata.py`, `1_run_detector.py`, `2_make_sample.py`, `5_score.py`.

## 1. What is evaluated

The cascade (`2_extraction/B_cascade`) decides for every crisis sentence whether it holds an **analogy**: the
TARGET sentence sets the present beside a past from before the crisis began. It does so in steps:

```
every sentence ──► 1 SCREEN  "does the TARGET set the present beside an earlier past?" (lenient)
                     │ yes                                   no ──► not counted
                     ▼
                   2a VERIFY "what is it set beside, and when?"  ─┐  analogy only if
                   2b CHECK  "is a remembered past a yardstick?" ─┤  both say yes
                     │ both yes                                   │  AND the screen's quote
                     ▼                                            │  lies in the TARGET
                   quote in TARGET? ──────────────────────────────┘
                     │ yes
                     ▼
                   COUNTED ──► 3 LABEL direction (similarity / rupture), kind of past
```

Every rule of the coders' question (`CODEBOOK.md`) matches the cascade's own definition: the "no" cases (the
crisis's own course, the future, the normal or usual, another place now, a past only mentioned, a comparison only
in a neighbouring sentence) are the verifier's rejection categories `within_crisis`, `future`, `routine`,
`elsewhere_now`, `not_compared` and the in-TARGET rule. So we measure how well the cascade does what it was
designed to do, not a different notion of analogy.

**Quantities** (with 95% intervals):

| quantity | meaning |
|---|---|
| precision | share of counted analogies that are real |
| recall | share of real analogies that are counted |
| true rate | real analogies per 1,000 sentences |
| per era | precision, recall and true rate for crises before 1940, 1940–69 and 1970–92 |
| losses | real analogies lost at each step: screen, verifiers, quote check |
| labels | how often direction and kind are right, for correctly counted analogies |
| agreement | how far the three coders agree; how far the model agrees with them |

## 2. Population: the test sentences

The main run covers all 139,344 crisis sentences. **23,408 were used while building the cascade** and are left
out: the samples of earlier runs (`run1_2000`, `run2_5000`, `model26b_1000`), the ids in
`excluded_ids_earlier_runs.txt`, and every sentence that appears as a few-shot example. On these the cascade was
tuned, so they would flatter it. That leaves **N = 115,936 test sentences**.

## 3. Strata: one per decision of the cascade

Analogies are rare (about 2.5% of sentences): 1,000 random sentences would hold only some 25. So we do not sample
at random. We sort every test sentence by **what the cascade decided** and sample each group separately. Each
group corresponds to one point where the cascade can be wrong:

| stratum | cascade decision | can be wrong by | sentences | coded | coders |
|---|---|---|---|---|---|
| A_pre1940, A_1940_1969, A_1970_1992 | counted (split by era of the crisis) | false positive → **precision** | 388 / 791 / 1,727 | 50 each | 2 |
| B_split | screen yes, one verifier yes, one no | the "both yes" rule loses it | 1,708 | 70 | 2 |
| C_rejected | screen yes, both verifiers no | both verifiers wrongly reject | 2,985 | 50 | 2 |
| D_neighbour | both verifiers yes, quote not in TARGET | the quote check wrongly rejects | 25 | all | 2 |
| Z_original | screen no, but the original pipeline (A) found a comparison in the TARGET | screen miss | 90 | 50 | 2 |
| E1_flag / E1_noflag | screen no, past cue in the sentence; detector yes / no | screen miss | 803 / 8,209 | 45 / 110 | 2 / 1 |
| E2_flag / E2_noflag | screen no, no past cue; detector yes / no | screen miss | 2,978 / 96,232 | 45 / 420 | 2 / 1 |

**Allocation.** E2_noflag is by far the largest stratum, and with an expected share of analogies of roughly
0.1–0.6% it carries about 60% of the uncertainty in the total number of real analogies. It therefore gets the most
items (420; the Neyman-optimal allocation for the total would give it about the same share). More would barely
help: 650 items instead of 420 narrows the recall interval by only about one more point (simulated). The A strata
are kept at 50 per era, more than the total alone would need, because precision per era is a main result.

Together the strata cover every test sentence exactly once. Every real analogy the cascade misses therefore sits
in B, C, D, Z or E, and the stratum says at which step it was lost.

*Past cue* (E1 vs E2): a regular expression for words that often signal a past (eerder, vroeger, destijds, ooit,
nooit, opnieuw, weer, sinds, geleden, net als, vorige, a year 18xx/19xx, …; `strata.CUE`).

### The detector and two-phase sampling (E strata)

The screen-no strata hold 93% of the test sentences and almost no analogies, yet they decide recall: in a
stratum of 99,000 sentences, one real analogy among a few hundred coded items stands for hundreds. To find the
few that are there, `1_run_detector.py` asks a **second, independent opinion**: the cascade's `check` question,
zero-shot (without the cascade's few-shot examples), put to a different model (Qwen3-14B instead of Gemma-4 12B).
It ran for 30 minutes over a shuffled list of E1 and E2 sentences; the 7,894 sentences it finished are a random
**phase-1 sample** (2,963 of E1, 4,931 of E2). It flagged 412 of them (8.9% of E1, 3.0% of E2).

The size of each sub-stratum is then *estimated* from phase 1:

  N(E2_flag) = N(E2) × m(E2_flag) / m(E2)

where m counts phase-1 sentences. The coded items are drawn from the phase-1 sentences of each sub-stratum.
This is standard two-phase (double) sampling: the estimates stay unbiased **whatever the detector's quality**;
a good detector only makes them more precise. The uncertainty of the estimated sizes is carried into the
intervals (section 6).

## 4. Coding

- **Three coders** (Ruben, Fien, Adriaan), **one list each** (about 467 sentences), one round.
- **Double coding** (A, B, C, D, Z, E*_flag: 435 items): two coders per item. The three pairs rotate within
  every stratum, so each pair shares about 145 items with the same mix of strata.
- **Single coding** (E1_noflag, E2_noflag: 530 items): these are almost all plain "no", and double coding them
  would halve the sample where precision matters most.
- **Blind**: coders see the crisis, the article context and the TARGET sentence, never the stratum or the
  model's answers (`key.csv` stays local). Items are mixed in random order per coder.
- **Question** (`CODEBOOK.md`): analogy yes/no; if yes, direction (similarity / rupture), kind of past
  (named_event / series / whole_past / general_period) and the past referred to; *unsure* allowed. The codebook
  is frozen once coding starts; every answer stores its version.

### Gold label per item

1. The adjudicator's decision, if there is one.
2. Otherwise the coders' agreed answer (double-coded), or the single coder's answer.

The adjudicator (Ruben) decides every double-coded item where the two disagree (on analogy, or on direction or
kind when both say yes), every item marked *unsure*, and every **single-coded yes**: in E2_noflag a single false
"yes" would count for hundreds of missed analogies, so none goes unchecked. The adjudicator sees the answers
without names and never the model's. Re-checking only items where humans disagree *with the model* would bias
the result towards the model; the trigger here is the coders' own answers.

Until adjudication, a disagreement counts as ½ and a single-coded "yes" as 1 (the report marks them provisional).

## 5. Point estimates

For each stratum h: N_h sentences (estimated for E sub-strata), n_h coded items, gold labels y_hi ∈ {0, ½, 1}.

| | formula |
|---|---|
| share of real analogies in h | p_h = Σ_i y_hi / n_h |
| real analogies in h | T_h = N_h · p_h |
| counted by the cascade | M = Σ_{h ∈ A} N_h |
| correctly counted | TP = Σ_{h ∈ A} T_h |
| all real analogies | T = Σ_h T_h |
| **precision** | TP / M |
| **recall** | TP / T |
| **true rate per 1,000** | 1,000 · T / N |
| F1 | 2 · precision · recall / (precision + recall) |
| lost at a step | Σ T_h over that step's strata (B: verifiers disagree; C: both verifiers no; D: quote check; Z + E: screen) |

This is the same as weighting every coded item by w_h = N_h / n_h (the sentences it stands for).

**Per era.** Every sentence belongs to one crisis and so to one era. The A strata are already per era. For the
other strata, the real analogies are split over eras by the eras of the coded positives:
T_{h,e} = N_h · Σ_{i ∈ h, era e} y_hi / n_h. Then precision_e = TP_e / M_e, recall_e = TP_e / T_e and
rate_e = 1,000 · T_e / N_e, with N_e the test sentences of that era.

**Correcting the cascade's rates.** A counted rate r (per crisis, era, newspaper, …) estimates a true rate of
r · precision / recall. Rates can be compared across eras as they are only if precision/recall is about equal
across eras; otherwise correct per era first.

## 6. Intervals

`5_score.py` draws 20,000 times from the posterior of every unknown and reports the 2.5th and 97.5th percentiles:

- **share in each stratum:** p_h ~ Beta(k_h + ½, n_h − k_h + ½), with k_h = Σ y_hi (Jeffreys prior);
- **era split of a stratum's positives:** Dirichlet(positives per era + the era mix of the coded items);
- **sizes of the E sub-strata:** shares ~ Dirichlet(m_flag + ½, m_noflag + ½) from the phase-1 counts.

Each draw gives precision, recall, rate and losses by the formulas of section 5. Unlike a bootstrap, a stratum
with zero positives still adds uncertainty: 0 of 420 in E2_noflag does not mean 0 analogies in 96,232 sentences,
and recall's lower bound reflects that. Per-stratum shares are reported with the same Jeffreys intervals.

## 7. Agreement

On the 435 double-coded items:

- **Krippendorff's alpha** (nominal) for analogy yes/no, with a bootstrap interval over items; raw agreement.
- **Cohen's kappa per pair**: does one coder stand apart?
- **Positive specific agreement**, 2a / (2a + b + c): how often one coder's "yes" is shared by the other.
  Kappa can look low on rare categories; this is the direct figure for the "yes" judgements that matter.
- **Kappa weighted to the corpus** (items weighted by w_h): the sample is enriched with hard cases, so
  unweighted agreement describes the sample, not the corpus.
- **The model as a coder**: kappa between the cascade's decision and each human, next to kappa between the
  humans. If they are close, the cascade's errors are of the size of human disagreement.
- Direction and kind: alpha and share of identical answers where both coders say yes.

## 8. Robustness

Precision, recall and the true rate are recomputed with other gold labels:

- **strict**: yes only if every coder (and the adjudicator, if any) said yes;
- **lenient**: yes if anyone did;
- **each coder alone**, on the items that coder coded. "Fien alone" and "Adriaan alone" do not depend on the
  adjudicator, who is also a coder.

Conclusions that hold in every row do not depend on how disagreements are settled or on who coded.

## 9. Labels

On counted analogies that are real (A strata, gold yes), weighted by w_h: the share where the cascade's
direction and kind equal the gold ones, and a confusion table for kind.

## 10. Limits

- In the single-coded strata a coder's false "no" is not caught. Its effect is at most the coder's miss rate
  (visible on the double-coded items) times the few real analogies there.
- The adjudicator is also a coder; section 8 shows whether that matters.
- Per crisis the sample is too small; era is the finest level with usable intervals.
- Not evaluated here: whether the quoted span is the right one, and the named events (names, years, types),
  which are checked in the gazetteer (`3_named_events/`).
