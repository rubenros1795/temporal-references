"""
Copy the coders' labels from the GitHub `labels` branch (where the Streamlit Cloud app saves them) into labels/.

    .venv_annotator/bin/python evaluation/3_pull_labels.py --repo rubenros1795/temporal-references

Uses the token of the GitHub CLI (`gh auth token`) or $GITHUB_TOKEN. Overwrites labels/<coder>.jsonl, never
labels/_review.jsonl (the adjudicator's decisions, which live only here).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from labels_io import LABELS, GitHubStore, label_file, parse  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="rubenros1795/temporal-references")
    ap.add_argument("--branch", default="labels")
    args = ap.parse_args()
    token = os.environ.get("GITHUB_TOKEN") or subprocess.run(["gh", "auth", "token"], capture_output=True,
                                                             text=True).stdout.strip()
    gh = GitHubStore(token, args.repo, args.branch)
    LABELS.mkdir(exist_ok=True)
    for coder in json.load(open(HERE / "data" / "setup.json", encoding="utf-8"))["coders"]:
        text, _ = gh._get(coder)
        label_file(coder).write_text(text, encoding="utf-8")
        print(f"{coder}: {len(parse(text))} items coded")


if __name__ == "__main__":
    main()
