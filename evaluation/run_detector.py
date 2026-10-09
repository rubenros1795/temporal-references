"""
Independent detector for the screen-no sentences, used only to split strata E1 and E2 for sampling.

The screen said no to these sentences, so the cascade's verifiers never saw them. Here the cascade's `check`
question is asked of a phase-1 sample of E1 (screen no, past cue) and E2 (screen no, no cue). Sentences it
calls an analogy form E1_flag / E2_flag, the rest E1_noflag / E2_noflag. The detector only decides how densely
each part is sampled; the weights keep every estimate unbiased whatever its quality.

The phase-1 list is shuffled, so the run can be stopped at any time (e.g. `timeout 30m`): the sentences
finished by then are a random sample of the list, and make_sample.py uses exactly those.

--zero-shot drops the cascade's few-shots (about 12k tokens), keeping the question and its definitions. With a
different model (--url pointing at another server) this gives a detector that shares little with the screen.

    timeout 30m .venv_annotator/bin/python evaluation/run_detector.py --url http://127.0.0.1:8082 --zero-shot \
        --model-name "Qwen3-14B" > logs/eval_detector.log 2>&1
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata  # noqa: E402  (also puts the project root and B_cascade on sys.path)
import config  # noqa: E402
import prompts  # noqa: E402
from schemas import SCHEMAS  # noqa: E402

sys.path.insert(0, str(config.PROJECT_DIR.parent / "llm-annotation"))
from llm_annotation import annotate_batch  # noqa: E402

COLS = ["instance_id", "event", "article_id", "date", "sentence_index"]
INFO = strata.RUN / "eval_detector_info.json"


def zero_shot_messages(row):
    p = prompts.payload(row["date"], row["event"], *prompts.window(row["article_id"], int(row["sentence_index"])))
    return [{"role": "system", "content": prompts._system("check")},
            {"role": "user", "content": "INPUT:\n" + json.dumps(p, ensure_ascii=False) + "\nOUTPUT:"}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", type=int, default=9012, help="phase-1 list size from E1 (all by default)")
    ap.add_argument("--e2", type=int, default=15000, help="phase-1 list size from E2")
    ap.add_argument("--url", default=config.LLAMA_SERVER_URL)
    ap.add_argument("--zero-shot", action="store_true")
    ap.add_argument("--model-name", default=config.MODEL_PRESET, help="only recorded, for the report")
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if strata.DETECTOR_MANIFEST.exists():
        man = pd.read_csv(strata.DETECTOR_MANIFEST, dtype=str)
        print(f"resuming: {len(man):,} sentences in the existing list", file=sys.stderr)
    else:
        d = strata.load_main()
        t = d[~d.dev]
        parts = [t[t.parent == p].sample(min(n, (t.parent == p).sum()), random_state=args.seed)
                 for p, n in [("E1_cue", args.e1), ("E2_nocue", args.e2)]]
        man = pd.concat(parts)[COLS + ["parent"]].sample(frac=1, random_state=args.seed)   # shuffled: stop any time
        man.to_csv(strata.DETECTOR_MANIFEST, index=False)
        json.dump({"model": args.model_name, "url": args.url, "prompt": "check, zero-shot" if args.zero_shot else "check"},
                  open(INFO, "w"), indent=2)
        print(f"phase-1 list: {len(parts[0]):,} E1 + {len(parts[1]):,} E2, shuffled", file=sys.stderr)

    model, schema = SCHEMAS["check"]
    build = zero_shot_messages if args.zero_shot else (lambda r: prompts.messages("check", r))
    n, err, tripped = annotate_batch(man[COLS].to_dict("records"), build, model, schema, str(strata.DETECTOR_JSONL),
                                     args.url, workers=args.workers, confidence_field=None)
    if tripped:
        sys.exit(3)
    print(f"done: {n:,} annotated, {err:,} errors", file=sys.stderr)


if __name__ == "__main__":
    main()
