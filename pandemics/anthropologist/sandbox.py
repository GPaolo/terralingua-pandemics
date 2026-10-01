"""Guarded executor for model-written analysis code.

Layers, in order: an AST screen rejects imports outside ALLOWED_ROOTS, dunder
attribute access and eval/exec-style names before anything runs; the code then
executes in a separate worker process with a per-call wall-clock timeout, a
memory cap where the OS honours it, file reads confined to the run, the
package, the interpreter, system fonts and the temp dir, and writes confined
to the chat plots dir, the temp dir and matplotlib's own cache.
Nothing network-capable is importable.

These layers stop accidents and log-borne prompt injection, but Python cannot
be fully sandboxed in-language — the approve-before-run gate in the chat and
dashboard is the actual security boundary. Keep it on for untrusted runs.
"""

import ast
import builtins
import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pandemics.anthropologist import epidemic_utils  # noqa: E402

try:
    import resource
except ImportError:  # not on this platform
    resource = None

PACKAGE_DIR = Path(__file__).resolve().parents[1]

ALLOWED_ROOTS = {
    "math", "statistics", "itertools", "functools", "collections", "json",
    "re", "csv", "random", "datetime", "textwrap", "heapq", "bisect",
    "pathlib", "numpy", "pandas", "matplotlib", "epidemic_utils",
}

BANNED_NAMES = {
    "eval", "exec", "compile", "__import__", "getattr", "setattr", "delattr",
    "globals", "vars", "breakpoint", "input", "exit", "quit", "memoryview",
}

TIMEOUT = 30
MAX_OUTPUT = 20000


def check(code: str) -> list:
    """Violations that make the code unrunnable ([] = clean)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_ROOTS:
                    problems.append(f"import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in ALLOWED_ROOTS:
                problems.append(f"import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            problems.append(f"'{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            problems.append(f"dunder attribute '.{node.attr}' is not allowed")
    return sorted(set(problems))


class Sandbox:
    """Persistent worker process; run() keeps state between calls."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir).resolve()
        self.proc = None
        self._lock = threading.Lock()
        self._spawn()

    def _spawn(self):
        self.proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "pandemics.anthropologist.sandbox", "--worker", str(self.run_dir)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        assert self.proc.stdout.readline().strip() == "ready"

    def run(self, code: str, timeout: int = TIMEOUT) -> str:
        problems = check(code)
        if problems:
            return "Blocked by the sandbox, nothing ran:\n- " + "\n- ".join(problems)
        with self._lock:
            if self.proc.poll() is not None:
                self._spawn()
            self._active = True
            self.proc.stdin.write(json.dumps({"code": code, "timeout": timeout}) + "\n")
            self.proc.stdin.flush()
            reply = [None]
            reader = threading.Thread(
                target=lambda: reply.__setitem__(0, self.proc.stdout.readline()),
                daemon=True,
            )
            reader.start()
            reader.join(timeout + 10)
            self._active = False
            if reply[0] is None or not reply[0].strip():
                self.proc.kill()
                self._spawn()
                return ("Killed (timeout, interrupt, or worker death). The "
                        "sandbox restarted: session variables are gone.")
            return json.loads(reply[0])["output"]

    def interrupt(self):
        """Kill an in-flight call; a no-op when idle."""
        if getattr(self, "_active", False) and self.proc.poll() is None:
            self.proc.kill()

    def reset(self):
        with self._lock:
            self.close()
            self._spawn()

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()


# ---------------------------------------------------------------- worker side

def _worker(run_dir: Path):
    if resource is not None:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
        except (ValueError, OSError):
            pass

    # Model code says `import epidemic_utils`; the package module answers to it.
    sys.modules["epidemic_utils"] = epidemic_utils
    eu = epidemic_utils

    chat_dir = run_dir / "epidemic_analysis" / "chat"
    chat_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.gettempdir())
    read_ok = [run_dir, PACKAGE_DIR, Path(sys.prefix), Path(sys.base_prefix),
               Path.home() / ".matplotlib", Path("/System/Library/Fonts"),
               Path("/Library/Fonts"), tmp]
    write_ok = [chat_dir, tmp, Path.home() / ".matplotlib"]

    def _inside(path, roots):
        try:
            p = Path(path).resolve()
        except (OSError, ValueError):
            return False
        return any(p == r or p.is_relative_to(r) for r in roots)

    def _guard(path, write):
        roots = write_ok if write else read_ok
        if not _inside(path, roots):
            kind = "write" if write else "read"
            raise PermissionError(
                f"sandbox: {kind} of {path} denied "
                + (f"(plots go under {chat_dir})" if write
                   else "(outside the run and the package)")
            )

    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if not isinstance(file, int):
            _guard(file, any(c in str(mode) for c in "wax+"))
        return real_open(file, mode, *args, **kwargs)

    real_os_open = os.open
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND

    def guarded_os_open(path, flags, *args, **kwargs):
        _guard(path, bool(flags & write_flags))
        return real_os_open(path, flags, *args, **kwargs)

    def denied(name):
        def _blocked(*args, **kwargs):
            raise PermissionError(f"sandbox: os.{name} is disabled")
        return _blocked

    proto_out = sys.stdout
    ns = {"RUN": run_dir, "eu": eu, "np": np, "json": json, "Path": Path,
          "plt": plt}

    def on_alarm(signum, frame):
        raise TimeoutError("sandbox: call timed out")

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, on_alarm)

    # Patch after all imports so library setup (font caches etc.) is unaffected.
    builtins.open = guarded_open
    io.open = guarded_open
    os.open = guarded_os_open
    for name in ("remove", "unlink", "rmdir", "removedirs", "rename",
                 "replace", "truncate", "system", "popen", "execv", "execve",
                 "fork", "kill"):
        if hasattr(os, name):
            setattr(os, name, denied(name))

    print("ready", file=proto_out, flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        buf = io.StringIO()
        try:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(int(req.get("timeout", TIMEOUT)))
            with contextlib.redirect_stdout(buf):
                exec(compile(req["code"], "<analysis>", "exec"), ns)
        except BaseException:
            traceback.print_exc(limit=6, file=buf)
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
        out = buf.getvalue() or "(no output — use print())"
        if len(out) > MAX_OUTPUT:
            out = out[:MAX_OUTPUT] + "\n... [truncated — print less]"
        print(json.dumps({"output": out}), file=proto_out, flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _worker(Path(sys.argv[2]).resolve())
    else:
        sys.exit("internal worker entry point; use Sandbox() from Python")
