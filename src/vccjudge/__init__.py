"""vccjudge -- the offline judge for VCC 2026 (Phase 0+)."""

# Submodules are NOT eagerly imported here. `python -m vccjudge.contract` is the
# readiness gate CLAUDE.md mandates, and an eager `from . import contract` makes
# runpy execute that module twice -- once as `vccjudge.contract` via this file,
# then again as `__main__` -- which Python warns is unpredictable. Callers use
# the explicit form: `from vccjudge.contract import validate_submission`.
__all__ = ["contract", "gene_axis", "synth"]
