"""Terminal anthropologist for a viral run (dashboard.py is the richer UI).

Usage:
    python scenarios/ebola_simulation/anthropologist/chat.py logs/<exp_name>
    (--model claude-opus-5 by default; needs ANTHROPIC_API_KEY or a .env)

Model-written code runs in the sandbox (sandbox.py) and, unless --auto-run is
given, only after you approve each snippet.
"""

import argparse
import readline  # noqa: F401  (line editing in input())
import sys
from pathlib import Path

import anthropic

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent  # noqa: E402
from sandbox import Sandbox  # noqa: E402

DENIED = ("The user declined to run this code. Explain what it would have "
          "done, or try a different approach.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="logs/<exp_name>")
    parser.add_argument("--model", default=agent.DEFAULT_MODEL)
    parser.add_argument("--auto-run", action="store_true",
                        help="run model code without asking (trusted runs only)")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not (run_dir / "world_state.jsonl").exists():
        sys.exit(f"{run_dir} does not look like a run directory (no world_state.jsonl)")

    sandbox = Sandbox(run_dir)
    system = agent.build_system(run_dir)
    client = anthropic.Anthropic()
    messages = []

    def executor(code):
        print("  [python]")
        print("\n".join(f"    {line}" for line in code.splitlines()))
        if not args.auto_run:
            try:
                if input("  run this? [y/N] ").strip().lower() not in ("y", "yes"):
                    return DENIED
            except (EOFError, KeyboardInterrupt):
                return DENIED
        out = sandbox.run(code)
        print("\n".join(f"    | {line}" for line in out.splitlines()[:15]))
        return out

    mode = "auto-run ON" if args.auto_run else "each snippet needs your approval"
    print(f"Anthropologist on {run_dir.name} ({args.model}; refusal fallback to "
          f"claude-opus-4-8 enabled; {mode}). Ask about the run; 'exit' to leave.")

    while True:
        try:
            question = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        checkpoint = len(messages)
        try:
            last = agent.run_turn(client, args.model, system, messages,
                                  question, executor,
                                  on_text=lambda t: print(f"\n{t}\n"),
                                  on_tool=lambda t: print(f"  [{t}]"))
            if last is not None and last.stop_reason == "refusal":
                print("  [the model declined this question]")
        except KeyboardInterrupt:
            del messages[checkpoint:]
            print("\n  [interrupted — this turn was discarded]")
        except anthropic.APIConnectionError:
            print("  [network error — check your connection and re-ask]")
        except anthropic.APIStatusError as e:
            print(f"  [API error {e.status_code}: {e.message} — turn dropped, re-ask]")
    sandbox.close()


if __name__ == "__main__":
    main()
