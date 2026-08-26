"""Live and replay dashboard for TerraLingua experiments.

Reads only the JSON/JSONL files an experiment writes into ``logs/<exp_name>/``.
Nothing here opens a pickle: ``env_state.pkl`` and ``checkpoint_latest.pkl`` hold
only the final state anyway, and unpickling a run downloaded from elsewhere would
execute arbitrary code.
"""
