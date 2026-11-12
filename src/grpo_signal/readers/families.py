"""Map rollouts onto task families from ``(env, task_key)``."""

from __future__ import annotations

import csv
import fnmatch
import io
import json
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, assert_never, cast

from grpo_signal.types import FamilySpec, Rollout

_GLOB_META = frozenset("*?[")
_DEFAULT_SEPARATOR = ":/"
_STRUCTURED_EXACT = "exact"
_STRUCTURED_GLOBS = "globs"


class FamilyAssigner(Protocol):
    """Pure ``(env, task_key) -> family`` callable."""

    def __call__(self, env: str, task_key: str) -> str: ...


class FamilyConfigError(ValueError):
    """Invalid family-mapping configuration."""


@dataclass(frozen=True, kw_only=True, slots=True)
class FamilyConfig:
    """How to map ``(env, task_key)`` to a family name.

    ``env`` (default): family is the environment name.
    ``prefix``: first ``n_parts`` of ``task_key`` split on ``separator``.
    ``regex``: ``pattern`` applied to ``task_key``; unmatched keys use
    ``fallback`` (``None`` means the environment name).
    ``mapping``: TOML or JSON file of exact keys and/or glob patterns.
    """

    mode: Literal["env", "prefix", "regex", "mapping"] = "env"
    n_parts: int = 1
    separator: str = _DEFAULT_SEPARATOR
    prefix_env: bool = False
    pattern: str | None = None
    fallback: str | None = None
    mapping_path: Path | None = None


def from_config(config: FamilyConfig | FamilySpec) -> FamilyAssigner:
    """Validate ``config`` and return a deterministic family assigner."""
    if isinstance(config, FamilySpec):
        return _from_family_spec(config)
    return _from_family_config(config)


def assign_family(r: Rollout, spec: FamilySpec) -> str:
    """Return the family name for a rollout under ``spec``."""
    return from_config(spec)(r.env, r.task_key)


def _from_family_spec(spec: FamilySpec) -> FamilyAssigner:
    if spec.kind == "env":
        return _from_family_config(FamilyConfig(mode="env"))
    if spec.kind == "regex":
        return _from_family_config(FamilyConfig(mode="regex", pattern=spec.pattern))
    if spec.kind == "csv":
        return _from_family_config(
            FamilyConfig(mode="mapping", mapping_path=spec.csv_path)
        )
    if spec.kind == "name":
        return _assign_task_key
    raise FamilyConfigError(
        f"FamilySpec kind {spec.kind!r} is not supported; "
        "assign_family maps (env, task_key) only"
    )


def _from_family_config(config: FamilyConfig) -> FamilyAssigner:
    if config.mode == "env":
        return _assign_env
    if config.mode == "prefix":
        return _prefix_assigner(config)
    if config.mode == "regex":
        return _regex_assigner(config)
    if config.mode == "mapping":
        return _mapping_assigner(config)
    assert_never(config.mode)


def _assign_env(env: str, task_key: str) -> str:
    return env


def _assign_task_key(env: str, task_key: str) -> str:
    return task_key


def _prefix_assigner(config: FamilyConfig) -> FamilyAssigner:
    if config.n_parts < 1:
        raise FamilyConfigError("prefix n_parts must be >= 1")
    n_parts = config.n_parts
    separator = config.separator
    prefix_env = config.prefix_env

    def assign(env: str, task_key: str) -> str:
        prefix = _task_key_prefix(task_key, n_parts, separator)
        if prefix_env:
            return _with_env_prefix(env, prefix, separator)
        return prefix

    return assign


def _task_key_prefix(task_key: str, n_parts: int, separator: str) -> str:
    sep = set(separator)
    found = 0
    start: int | None = None
    end = 0
    i = 0
    length = len(task_key)
    while i < length and found < n_parts:
        while i < length and task_key[i] in sep:
            i += 1
        if i >= length:
            break
        if start is None:
            start = i
        while i < length and task_key[i] not in sep:
            i += 1
        found += 1
        end = i
    if start is None:
        return ""
    return task_key[start:end]


def _with_env_prefix(env: str, prefix: str, separator: str) -> str:
    if not prefix:
        return env
    join = _env_join_char(separator)
    return f"{env}{join}{prefix}"


def _env_join_char(separator: str) -> str:
    if "/" in separator:
        return "/"
    if ":" in separator:
        return ":"
    if separator:
        return separator[0]
    return "/"


def _regex_assigner(config: FamilyConfig) -> FamilyAssigner:
    if config.pattern is None:
        raise FamilyConfigError("regex mode requires a pattern")
    try:
        compiled = re.compile(config.pattern)
    except re.error as exc:
        raise FamilyConfigError(f"invalid family regex: {exc}") from exc
    if "family" not in compiled.groupindex and compiled.groups < 1:
        raise FamilyConfigError(
            "family regex must contain a named group 'family' "
            "or at least one capturing group"
        )
    fallback = config.fallback

    def assign(env: str, task_key: str) -> str:
        match = compiled.search(task_key)
        if match is None:
            return _regex_fallback(env, fallback)
        if "family" in match.re.groupindex:
            value = match.group("family")
        else:
            value = match.group(1)
        if value is None or value == "":
            return _regex_fallback(env, fallback)
        return value

    return assign


def _regex_fallback(env: str, fallback: str | None) -> str:
    if fallback is None:
        return env
    return fallback


def _mapping_assigner(config: FamilyConfig) -> FamilyAssigner:
    if config.mapping_path is None:
        raise FamilyConfigError("mapping mode requires mapping_path")
    exact, globs = _load_mapping_file(config.mapping_path)

    def assign(env: str, task_key: str) -> str:
        hit = _lookup_mapping(task_key, exact, globs)
        if hit is None:
            return env
        return hit

    return assign


def _lookup_mapping(
    task_key: str,
    exact: Mapping[str, str],
    globs: Sequence[tuple[str, str]],
) -> str | None:
    if task_key in exact:
        return exact[task_key]
    for pattern, family in globs:
        if fnmatch.fnmatchcase(task_key, pattern):
            return family
    return None


def _load_mapping_file(path: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FamilyConfigError(
            f"cannot read family mapping file {path}: {exc}"
        ) from exc
    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = _parse_json_mapping(text, path)
    elif suffix in {".toml", ".tml"}:
        raw = _parse_toml_mapping(text, path)
    elif suffix == ".csv":
        return _items_to_exact_globs(_parse_csv_mapping(text, path))
    else:
        raw = _parse_mapping_unknown_suffix(text, path)
    return _mapping_object_to_exact_globs(raw, path)


def _parse_mapping_unknown_suffix(text: str, path: Path) -> object:
    try:
        return cast(object, json.loads(text))
    except json.JSONDecodeError:
        pass
    try:
        return cast(object, tomllib.loads(text))
    except tomllib.TOMLDecodeError:
        pass
    raise FamilyConfigError(f"family mapping file {path} is not valid JSON or TOML")


def _parse_json_mapping(text: str, path: Path) -> object:
    try:
        return cast(object, json.loads(text))
    except json.JSONDecodeError as exc:
        raise FamilyConfigError(
            f"family mapping file {path} is not valid JSON: {exc}"
        ) from exc


def _parse_toml_mapping(text: str, path: Path) -> object:
    try:
        return cast(object, tomllib.loads(text))
    except tomllib.TOMLDecodeError as exc:
        raise FamilyConfigError(
            f"family mapping file {path} is not valid TOML: {exc}"
        ) from exc


def _parse_csv_mapping(text: str, path: Path) -> list[tuple[str, str]]:
    rows = list(csv.reader(io.StringIO(text)))
    items: list[tuple[str, str]] = []
    start = 0
    if rows:
        header = [cell.strip().lower() for cell in rows[0]]
        if header[:2] == ["task_key", "family"]:
            start = 1
    for index, row in enumerate(rows[start:], start=start + 1):
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) < 2:
            raise FamilyConfigError(
                f"family mapping file {path} row {index} must have task_key,family"
            )
        key, family = row[0].strip(), row[1].strip()
        if not key:
            raise FamilyConfigError(
                f"family mapping file {path} row {index} has an empty task_key"
            )
        items.append((key, family))
    return items


def _as_str_object_dict(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    typed = cast(dict[object, object], value)
    out: dict[str, object] = {}
    for key, item in typed.items():
        if not isinstance(key, str):
            return None
        out[key] = item
    return out


def _as_object_list(value: object) -> list[object] | None:
    if not isinstance(value, list):
        return None
    return list(cast(list[object], value))


def _mapping_object_to_exact_globs(
    raw: object, path: Path
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    data = _as_str_object_dict(raw)
    if data is None:
        raise FamilyConfigError(
            f"family mapping file {path} must be a table of string keys to family names"
        )
    exact_val = data.get(_STRUCTURED_EXACT)
    globs_val = data.get(_STRUCTURED_GLOBS)
    structured = _as_str_object_dict(exact_val) is not None or isinstance(
        globs_val, (dict, list)
    )
    if structured:
        return _structured_mapping(data, path)
    items: list[tuple[str, str]] = []
    for key, value in data.items():
        if not isinstance(value, str):
            raise FamilyConfigError(
                f"family mapping file {path} must map strings to family names"
            )
        items.append((key, value))
    return _items_to_exact_globs(items)


def _structured_mapping(
    data: Mapping[str, object], path: Path
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    exact_raw = data.get(_STRUCTURED_EXACT)
    globs_raw = data.get(_STRUCTURED_GLOBS)
    if exact_raw is None:
        exact_table: dict[str, object] = {}
    else:
        maybe_exact = _as_str_object_dict(exact_raw)
        if maybe_exact is None:
            raise FamilyConfigError(
                f"family mapping file {path} key 'exact' must be a table of strings"
            )
        exact_table = maybe_exact
    exact = _require_str_dict(exact_table, what="exact", path=path)
    if globs_raw is None:
        globs: list[tuple[str, str]] = []
    else:
        globs = _coerce_globs(globs_raw, path)
    return exact, globs


def _require_str_dict(
    value: Mapping[str, object], *, what: str, path: Path
) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(item, str):
            raise FamilyConfigError(
                f"family mapping file {path} key {what!r} "
                "must map strings to family names"
            )
        if key not in out:
            out[key] = item
    return out


def _coerce_globs(value: object, path: Path) -> list[tuple[str, str]]:
    as_dict = _as_str_object_dict(value)
    if as_dict is not None:
        return list(_require_str_dict(as_dict, what="globs", path=path).items())
    as_list = _as_object_list(value)
    if as_list is not None:
        return [
            _coerce_glob_row(row, path, index)
            for index, row in enumerate(as_list, start=1)
        ]
    raise FamilyConfigError(
        f"family mapping file {path} key 'globs' must be a table or an array"
    )


def _coerce_glob_row(row: object, path: Path, index: int) -> tuple[str, str]:
    as_dict = _as_str_object_dict(row)
    if as_dict is not None:
        pattern = as_dict.get("pattern")
        family = as_dict.get("family")
        if isinstance(pattern, str) and isinstance(family, str):
            return pattern, family
    as_list = _as_object_list(row)
    if (
        as_list is not None
        and len(as_list) >= 2
        and isinstance(as_list[0], str)
        and isinstance(as_list[1], str)
    ):
        return as_list[0], as_list[1]
    raise FamilyConfigError(
        f"family mapping file {path} globs[{index}] must have pattern and family"
    )


def _is_glob(key: str) -> bool:
    return any(char in _GLOB_META for char in key)


def _items_to_exact_globs(
    items: Sequence[tuple[str, str]],
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    exact: dict[str, str] = {}
    globs: list[tuple[str, str]] = []
    for key, family in items:
        if _is_glob(key):
            globs.append((key, family))
        elif key not in exact:
            exact[key] = family
    return exact, globs
