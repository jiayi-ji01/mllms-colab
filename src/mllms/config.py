"""YAML configuration loading shared by command-line entry points."""

from __future__ import annotations

from collections.abc import Iterable
import argparse
from pathlib import Path
import sys
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping with consistent validation."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise ValueError(f"YAML config must contain a mapping: {path}")
    return document


def config_section(document: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    """Return one nested mapping from a loaded configuration."""
    value: Any = document
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            joined = ".".join(keys)
            raise ValueError(f"Missing config section: {joined}")
        value = value[key]
    if not isinstance(value, dict):
        raise ValueError(f"Config section must be a mapping: {'.'.join(keys)}")
    return dict(value)


def parse_configured_args(
    parser: argparse.ArgumentParser,
    default_config: Path,
    section: tuple[str, ...],
) -> argparse.Namespace:
    """Apply one YAML section as argparse defaults; explicit CLI flags win."""
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=Path, default=default_config)
    known, _ = probe.parse_known_args(sys.argv[1:])
    values = config_section(load_yaml(known.config), section)
    actions = {action.dest: action for action in parser._actions}
    unknown = set(values) - actions.keys()
    if unknown:
        raise ValueError(
            f"Unknown fields in {'.'.join(section)}: {sorted(unknown)}"
        )
    defaults: dict[str, Any] = {"config": known.config}
    for name, value in values.items():
        action = actions[name]
        if value is not None and action.type is not None:
            value = action.type(value)
        defaults[name] = value
    parser.set_defaults(**defaults)
    return parser.parse_args()
