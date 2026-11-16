# Contributor instructions

TIAGo MPC provides control components for the TIAGo Pro right arm.
Experiments are concise usage examples that teach how to compose those components.

Read `README.md` for setup and project structure. For physical robot work, also
read `docs/robot.md` and `docs/safety.md`.

## Maintenance

- Keep the implementation small and direct; prefer plain functions and dataclasses.
- Treat code and docstrings as the canonical API documentation. Keep the README
  focused on setup, purpose and short examples; avoid duplicating API details.
- Keep examples easy to read and modify. Put reusable control behavior in the package.
- Describe current behavior; Git preserves development history.
- Keep generated runtime models, caches and experiment results out of Git.
- Use the checkout's own environment; host and container environments are separate.
- Preserve upstream attribution, model provenance and license notices.
- Do not commit or push unless asked.

## Checks

Run relevant checks with the checkout's environment:

```bash
uv run python -m pytest -q
uv run python -m mypy
```
