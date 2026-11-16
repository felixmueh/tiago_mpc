# random100: 100 simulated reaches

Run fixed start/goal pairs, then count arrivals and joint-position violations
from the saved motions.

```bash
uv run python -m experiments.random100.run
uv run python -m experiments.random100.analyze <output-directory>
```

Use `run --pairs 8990` for a single case or `run --dry-run` to inspect the
configuration. Runs go under `_expr_data/`; each case keeps its own record.
Failures keep their output and the next case continues. Ctrl-C stops the study.

Analysis prints arrival and violation totals and writes `analysis.json`.
A position violation means a recorded joint exceeds its raw limit by more than
1e-5 rad. Missing or invalid motions are counted as unavailable.
