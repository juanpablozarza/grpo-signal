from __future__ import annotations

from pathlib import Path

import pytest

from grpo_signal.readers.families import (
    FamilyConfig,
    FamilyConfigError,
    assign_family,
    from_config,
)
from grpo_signal.types import FamilySpec, Rollout


def _rollout(*, env: str, task_key: str) -> Rollout:
    return Rollout(
        env=env,
        task_key=task_key,
        task_idx=None,
        family="",
        group_id="g0",
        reward=0.0,
        rewards={},
        shipped=None,
        step=None,
    )


def _assign(config: FamilyConfig, env: str, task_key: str) -> str:
    return from_config(config)(env, task_key)


@pytest.mark.parametrize(
    ("env", "task_key", "expected"),
    [
        ("math", "algebra/easy/q1", "math"),
        ("code", "humaneval:0", "code"),
        ("gsm8k", "", "gsm8k"),
        ("env", "t0", "env"),
    ],
)
def test_env_mode(env: str, task_key: str, expected: str) -> None:
    assert _assign(FamilyConfig(), env, task_key) == expected
    assert _assign(FamilyConfig(mode="env"), env, task_key) == expected
    assert assign_family(_rollout(env=env, task_key=task_key), FamilySpec()) == expected


@pytest.mark.parametrize(
    ("env", "task_key", "n_parts", "separator", "prefix_env", "expected"),
    [
        ("e", "a/b/c", 1, ":/", False, "a"),
        ("e", "a/b/c", 2, ":/", False, "a/b"),
        ("e", "a/b/c", 3, ":/", False, "a/b/c"),
        ("e", "a/b/c", 8, ":/", False, "a/b/c"),
        ("e", "a:b:c", 2, ":/", False, "a:b"),
        ("e", "a/b:c/d", 2, ":/", False, "a/b"),
        ("e", "a/b:c/d", 3, ":/", False, "a/b:c"),
        ("e", "abc", 1, ":/", False, "abc"),
        ("e", "abc", 2, ":/", False, "abc"),
        ("e", "/a/b", 1, ":/", False, "a"),
        ("e", "a//b/c", 2, ":/", False, "a//b"),
        ("e", "a.b.c", 2, ".", False, "a.b"),
        ("e", "a-b-c", 1, "-", False, "a"),
        ("math", "algebra/easy/q1", 1, ":/", True, "math/algebra"),
        ("math", "algebra:easy:q1", 2, ":", True, "math:algebra:easy"),
        ("code", "humaneval.0", 1, ".", True, "code.humaneval"),
        ("env", "", 1, ":/", False, ""),
        ("env", "", 1, ":/", True, "env"),
        ("env", "only", 1, ":/", True, "env/only"),
    ],
)
def test_prefix_mode(
    env: str,
    task_key: str,
    n_parts: int,
    separator: str,
    prefix_env: bool,
    expected: str,
) -> None:
    config = FamilyConfig(
        mode="prefix",
        n_parts=n_parts,
        separator=separator,
        prefix_env=prefix_env,
    )
    assert _assign(config, env, task_key) == expected


@pytest.mark.parametrize(
    ("env", "task_key", "pattern", "fallback", "expected"),
    [
        ("e", "math/q1", r"(?P<family>math|code)/.*", None, "math"),
        ("e", "code/q1", r"(?P<family>math|code)/.*", None, "code"),
        ("e", "other/q1", r"(?P<family>math|code)/.*", None, "e"),
        ("e", "other/q1", r"(?P<family>math|code)/.*", "misc", "misc"),
        ("gsm8k", "train/12", r"(math|code)/.*", "misc", "misc"),
        ("gsm8k", "math/12", r"(math|code)/.*", "misc", "math"),
        ("e", "prefix-math-suffix", r"(.*)(?P<family>math)(.*)", None, "math"),
        ("e", "abc", r"(?P<family>a)(?P<rest>bc)", None, "a"),
        ("e", "zz", r"(?P<family>math)|zz", None, "e"),
        ("e", "math", r"math|(?P<family>code)", None, "e"),
    ],
)
def test_regex_mode(
    env: str,
    task_key: str,
    pattern: str,
    fallback: str | None,
    expected: str,
) -> None:
    config = FamilyConfig(mode="regex", pattern=pattern, fallback=fallback)
    assert _assign(config, env, task_key) == expected
    spec = FamilySpec(kind="regex", pattern=pattern)
    if fallback is None:
        assert assign_family(_rollout(env=env, task_key=task_key), spec) == expected


@pytest.mark.parametrize(
    ("text", "suffix"),
    [
        (
            '{"gsm8k/train/1": "gsm8k", "math/*": "math", "code*": "code"}',
            ".json",
        ),
        (
            '"gsm8k/train/1" = "gsm8k"\n"math/*" = "math"\n"code*" = "code"\n',
            ".toml",
        ),
    ],
)
def test_mapping_file_modes(tmp_path: Path, text: str, suffix: str) -> None:
    path = tmp_path / f"map{suffix}"
    path.write_text(text, encoding="utf-8")
    assign = from_config(FamilyConfig(mode="mapping", mapping_path=path))
    assert assign("e", "gsm8k/train/1") == "gsm8k"
    assert assign("e", "math/algebra/q1") == "math"
    assert assign("e", "code-eval/0") == "code"
    assert assign("env", "unmapped/key") == "env"


def test_mapping_exact_beats_glob(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    path.write_text(
        '{"abc": "exact", "a*": "glob", "*": "star"}',
        encoding="utf-8",
    )
    assign = from_config(FamilyConfig(mode="mapping", mapping_path=path))
    assert assign("e", "abc") == "exact"
    assert assign("e", "abz") == "glob"
    assert assign("e", "zzz") == "star"


def test_mapping_first_glob_wins(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    path.write_text(
        '{"a*": "first", "ab*": "second"}',
        encoding="utf-8",
    )
    assign = from_config(FamilyConfig(mode="mapping", mapping_path=path))
    assert assign("e", "abc") == "first"

    path.write_text(
        '{"ab*": "second", "a*": "first"}',
        encoding="utf-8",
    )
    assign = from_config(FamilyConfig(mode="mapping", mapping_path=path))
    assert assign("e", "abc") == "second"


def test_mapping_structured_exact_key_with_glob_chars(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    path.write_text(
        '{"exact": {"a*": "literal"}, "globs": {"a*": "glob"}}',
        encoding="utf-8",
    )
    assign = from_config(FamilyConfig(mode="mapping", mapping_path=path))
    assert assign("e", "a*") == "literal"
    assert assign("e", "ab") == "glob"


def test_mapping_structured_exact_and_globs(tmp_path: Path) -> None:
    path = tmp_path / "map.toml"
    path.write_text(
        """
[exact]
"math/special" = "special"

[[globs]]
pattern = "math/*"
family = "math"

[[globs]]
pattern = "*"
family = "other"
""",
        encoding="utf-8",
    )
    assign = from_config(FamilyConfig(mode="mapping", mapping_path=path))
    assert assign("e", "math/special") == "special"
    assert assign("e", "math/algebra") == "math"
    assert assign("e", "code/q") == "other"


def test_mapping_unmapped_falls_back_to_env(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    path.write_text('{"only": "kept"}', encoding="utf-8")
    assign = from_config(FamilyConfig(mode="mapping", mapping_path=path))
    assert assign("gsm8k", "missing") == "gsm8k"


def test_mapping_loaded_once(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    path.write_text('{"a": "fam"}', encoding="utf-8")
    assign = from_config(FamilyConfig(mode="mapping", mapping_path=path))
    path.write_text('{"a": "changed"}', encoding="utf-8")
    assert assign("e", "a") == "fam"


def test_family_spec_csv_mapping(tmp_path: Path) -> None:
    path = tmp_path / "families.csv"
    path.write_text("task_key,family\nexact-key,exact\nmath/*,math\n", encoding="utf-8")
    spec = FamilySpec(kind="csv", csv_path=path)
    assign = from_config(spec)
    assert assign("e", "exact-key") == "exact"
    assert assign("e", "math/q1") == "math"
    assert assign("env", "other") == "env"
    assert assign_family(_rollout(env="e", task_key="exact-key"), spec) == "exact"


def test_family_spec_name_uses_task_key() -> None:
    assign = from_config(FamilySpec(kind="name"))
    assert assign("env", "task/a") == "task/a"


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (
            FamilyConfig(mode="prefix", n_parts=0),
            "n_parts",
        ),
        (
            FamilyConfig(mode="regex"),
            "pattern",
        ),
        (
            FamilyConfig(mode="regex", pattern="[unterminated"),
            "invalid family regex",
        ),
        (
            FamilyConfig(mode="regex", pattern="abc"),
            "named group",
        ),
        (
            FamilyConfig(mode="regex", pattern="(?:noncapturing)"),
            "named group",
        ),
        (
            FamilyConfig(mode="mapping"),
            "mapping_path",
        ),
    ],
)
def test_config_errors(config: FamilyConfig, match: str) -> None:
    with pytest.raises(FamilyConfigError, match=match):
        from_config(config)


def test_unreadable_mapping_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"
    with pytest.raises(FamilyConfigError, match="cannot read family mapping file"):
        from_config(FamilyConfig(mode="mapping", mapping_path=path))


def test_invalid_json_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(FamilyConfigError, match="not valid JSON"):
        from_config(FamilyConfig(mode="mapping", mapping_path=path))


def test_invalid_toml_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("= nope", encoding="utf-8")
    with pytest.raises(FamilyConfigError, match="not valid TOML"):
        from_config(FamilyConfig(mode="mapping", mapping_path=path))


def test_mapping_non_object(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text('["math"]', encoding="utf-8")
    with pytest.raises(FamilyConfigError, match="table of string keys"):
        from_config(FamilyConfig(mode="mapping", mapping_path=path))


def test_family_spec_field_rejected() -> None:
    with pytest.raises(FamilyConfigError, match="field"):
        from_config(FamilySpec(kind="field", field="difficulty"))
