"""Run viewer for the pandemics scenario.

A FastAPI server and one static page. The page shows the grid with the food,
the beings and their health, the chat, the artifacts, the per-step charts and
the chain of infections. It follows a run while it is written and replays a
finished one.

The viewer reads only the files of a run folder: ``world_state.jsonl``,
``open_gridworld.log``, ``agent_logs/``, ``costs.csv``, ``run_status.json``
and ``params.json``. Start it with ``python -m pandemics.viewer``.
"""
