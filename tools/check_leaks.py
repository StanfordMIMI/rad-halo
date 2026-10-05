"""Block restricted data, identifiers and local paths from being committed.

Runs on the files passed as arguments (pre-commit) or on all tracked files (CI).
"""

import re
import subprocess
import sys
from pathlib import Path

DATA_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".xlsx", ".ipynb", ".parquet", ".pkl", ".npy"}
DATA_DIRS = ("predictions/", "derived/", "results/", "tests/fixtures/", "annotation/label_studio/")
MAX_BYTES = 1_000_000
LARGE_FILE_ALLOWED = ("results/", "predictions/")

PATTERNS = {
    "MIMIC de-identification token": re.compile(r"_{3}"),
    "MIMIC note id": re.compile(r"\b\d{8}-RR-\d+\b"),
    "absolute local path": re.compile(r"(/Users/|/home/)[A-Za-z]"),
    "API key": re.compile(
        r"(sk-ant-[A-Za-z0-9]|sk-(proj-)?[A-Za-z0-9_-]{20}|hf_[A-Za-z0-9]{20}|AIza[0-9A-Za-z_-]{20})"
    ),
}
# Files that legitimately contain one of the patterns.
ALLOWED = {
    "MIMIC note id": {"derived/few_shot_note_ids.csv"},
    "MIMIC de-identification token": set(),
    "absolute local path": set(),
    "API key": set(),
}
SELF = "tools/check_leaks.py"


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
    return out.stdout.split()


def check(path: str) -> list[str]:
    problems = []
    p = Path(path)
    if not p.is_file() or path == SELF:
        return problems
    if p.suffix in DATA_SUFFIXES and not path.startswith(DATA_DIRS):
        problems.append(f"{path}: data file outside {', '.join(DATA_DIRS)}")
    if p.stat().st_size > MAX_BYTES and not path.startswith(LARGE_FILE_ALLOWED):
        problems.append(f"{path}: larger than {MAX_BYTES} bytes")
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return problems
    for name, pattern in PATTERNS.items():
        if path in ALLOWED[name]:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                problems.append(f"{path}:{lineno}: {name}")
                break
    return problems


def main(argv: list[str]) -> int:
    files = argv or tracked_files()
    problems = [msg for f in files for msg in check(f)]
    for msg in problems:
        print(msg)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
