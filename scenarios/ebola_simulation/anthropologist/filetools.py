"""Read-only file navigation for the anthropologist (Read/Grep/LS style).

No code execution: plain host-side reads confined to the run directory and
the repo, so these need no sandbox and no approval. The ~12 KB input_prompt
fields are stripped from .jsonl lines before anything reaches the model.
Every function returns a user-facing string, errors included.
"""

import json
import re
from pathlib import Path

MAX_LINE = 500
MAX_OUTPUT = 20000
MAX_MATCHES = 50
MAX_ENTRIES = 200
GLOB_CHARS = set("*?[")


class Scope:
    """Paths resolve against the run dir first, then the repo root."""

    def __init__(self, run_dir, repo_root):
        self.run_dir = Path(run_dir).resolve()
        self.repo = Path(repo_root).resolve()

    def resolve(self, path):
        p = Path(path)
        candidates = [p] if p.is_absolute() else [self.run_dir / p, self.repo / p]
        for cand in candidates:
            r = cand.resolve()
            if r.is_relative_to(self.run_dir) or r.is_relative_to(self.repo):
                if r.exists():
                    return r
            else:
                raise PermissionError(f"{path} is outside the run and the repo")
        raise FileNotFoundError(f"{path} not found in the run or the repo")


def _clean(line, path):
    if path.suffix == ".jsonl" and '"input_prompt"' in line:
        try:
            row = json.loads(line)
            row["input_prompt"] = "…stripped…"
            line = json.dumps(row, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            pass
    line = line.rstrip("\n")
    if len(line) > MAX_LINE:
        line = line[:MAX_LINE] + f" …[+{len(line) - MAX_LINE} chars]"
    return line


def _is_binary(path):
    with open(path, "rb") as f:
        return b"\0" in f.read(1024)


def list_files(scope, path="."):
    try:
        if GLOB_CHARS & set(str(path)):
            pattern = str(path)
            entries = sorted(scope.run_dir.glob(pattern))
            if not entries:
                entries = sorted(scope.repo.glob(pattern))
        else:
            entries = sorted(scope.resolve(path).iterdir())
    except (PermissionError, FileNotFoundError, NotADirectoryError, OSError) as e:
        return str(e)
    lines = []
    for p in entries[:MAX_ENTRIES]:
        if p.is_dir():
            lines.append(f"{p.name}/  ({sum(1 for _ in p.iterdir())} entries)")
        else:
            lines.append(f"{p.name}  ({p.stat().st_size:,} bytes)")
    if len(entries) > MAX_ENTRIES:
        lines.append(f"… +{len(entries) - MAX_ENTRIES} more")
    return "\n".join(lines) or "(empty)"


def read_file(scope, path, offset=1, limit=200):
    try:
        p = scope.resolve(path)
        if p.is_dir():
            return f"{path} is a directory — use list_files"
        if _is_binary(p):
            return f"{path} is binary — not readable as text"
    except (PermissionError, FileNotFoundError, OSError) as e:
        return str(e)
    offset = max(1, int(offset))
    out, total = [], 0
    with open(p, errors="replace") as f:
        for n, line in enumerate(f, 1):
            if n < offset:
                continue
            if n >= offset + int(limit):
                out.append(f"… stopped at line {n - 1}; pass offset={n} for more")
                break
            text = f"{n}\t{_clean(line, p)}"
            total += len(text)
            if total > MAX_OUTPUT:
                out.append(f"… output cap hit at line {n}; pass offset={n} for more")
                break
            out.append(text)
    return "\n".join(out) or f"{path} is empty (or offset is past the end)"


def grep_files(scope, pattern, path=".", glob="**/*"):
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"bad regex: {e}"
    try:
        base = scope.resolve(path)
    except (PermissionError, FileNotFoundError) as e:
        return str(e)
    files = [base] if base.is_file() else sorted(
        p for p in base.glob(glob) if p.is_file()
    )
    out, matches, total = [], 0, 0
    for p in files:
        try:
            if _is_binary(p):
                continue
            with open(p, errors="replace") as f:
                for n, line in enumerate(f, 1):
                    line = _clean(line, p)
                    if not rx.search(line):
                        continue
                    matches += 1
                    rel = p.relative_to(base) if base.is_dir() else p.name
                    text = f"{rel}:{n}: {line[:300]}"
                    total += len(text)
                    out.append(text)
                    if matches >= MAX_MATCHES or total > MAX_OUTPUT:
                        out.append(f"… stopped at {matches} matches — narrow the "
                                   "pattern or glob")
                        return "\n".join(out)
        except OSError:
            continue
    return "\n".join(out) or f"no matches for /{pattern}/ in {path}/{glob}"
