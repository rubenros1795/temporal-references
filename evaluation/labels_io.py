"""Reading and writing coder labels: local files, or a GitHub branch when the app runs on Streamlit Cloud.

Labels are one JSON line per save in labels/<coder>.jsonl; the latest answer per item counts.
On Streamlit Cloud the disk is wiped on every restart, so app.py stores the same files on a branch of a
GitHub repo instead (GitHubStore). 3_pull_labels.py copies them back into labels/ for adjudication and scoring.
"""
import base64
import json
import re
import time
from pathlib import Path

import requests

LABELS = Path(__file__).resolve().parent / "labels"


def label_file(coder):
    return LABELS / f"{re.sub(r'[^A-Za-z0-9_-]', '_', coder)}.jsonl"


def parse(text):
    """item_id -> the latest answer, from the text of a labels file."""
    out = {}
    for line in text.splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["item_id"]] = r
    return out


def read_labels(path):
    return parse(path.read_text(encoding="utf-8")) if path.exists() else {}


def append(path, rec):
    path.parent.mkdir(exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class LocalStore:
    def read(self, coder):
        return read_labels(label_file(coder))

    def append(self, coder, rec):
        append(label_file(coder), rec)


class GitHubStore:
    """labels/<coder>.jsonl on a branch of a GitHub repo, through the contents API (one commit per save)."""

    def __init__(self, token, repo, branch="labels"):
        self.repo, self.branch = repo, branch
        self.h = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

    def _url(self, coder):
        return f"https://api.github.com/repos/{self.repo}/contents/labels/{label_file(coder).name}"

    def _get(self, coder):
        r = requests.get(self._url(coder), headers=self.h, params={"ref": self.branch}, timeout=20)
        if r.status_code == 404:
            return "", None
        r.raise_for_status()
        j = r.json()
        return base64.b64decode(j["content"]).decode("utf-8"), j["sha"]

    def read(self, coder):
        return parse(self._get(coder)[0])

    def append(self, coder, rec):
        for attempt in range(4):   # retry if the file changed in between (e.g. the same coder in two tabs)
            text, sha = self._get(coder)
            body = {"message": f"{coder}: {rec['item_id']}", "branch": self.branch,
                    "content": base64.b64encode((text + json.dumps(rec, ensure_ascii=False) + "\n").encode()).decode()}
            if sha:
                body["sha"] = sha
            r = requests.put(self._url(coder), headers=self.h, json=body, timeout=20)
            if r.status_code in (200, 201):
                return
            if r.status_code not in (409, 422):
                r.raise_for_status()
            time.sleep(1 + attempt)
        raise RuntimeError("Could not save to GitHub; please try again.")
