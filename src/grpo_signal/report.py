"""JSON, HTML, and terminal reports."""

from pathlib import Path

from grpo_signal.types import Report


def write_json(report: Report, path: Path) -> None:
    """Write ``report.json`` with the stable Report schema."""
    raise NotImplementedError("write_json")


def write_html(report: Report, path: Path) -> None:
    """Write a single static ``report.html`` file."""
    raise NotImplementedError("write_html")


def render_terminal(report: Report) -> str:
    """Return the short per-family table for the terminal."""
    raise NotImplementedError("render_terminal")
