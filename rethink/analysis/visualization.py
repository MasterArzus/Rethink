"""Visualization stubs for notebook or dashboard integrations."""

from __future__ import annotations

from typing import Iterable

from .structures import DivergenceReport


def render_prob_trajectory(report: DivergenceReport) -> None:
    """Placeholder for a Plotly/Matplotlib implementation."""

    raise NotImplementedError("Visualization layer to be implemented")


def render_hidden_state_heatmap(report: DivergenceReport) -> None:
    """Placeholder function for hidden-state diagnostics."""

    raise NotImplementedError("Visualization layer to be implemented")
