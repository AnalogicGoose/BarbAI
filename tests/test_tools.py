import os

import pytest

from barbai.core.tools import ToolExecutionError, read_file


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "hello.txt").write_text("The secret code is 4271.")
    sub = root / "sub"
    sub.mkdir()
    (sub / "nested.txt").write_text("nested content")

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "secret.txt").write_text("SHOULD NEVER BE READABLE")

    monkeypatch.setenv("BARBAI_TOOLS_ROOTS", str(root))
    return root


def test_valid_read(workspace):
    assert read_file("hello.txt") == "The secret code is 4271."


def test_valid_nested_read(workspace):
    assert read_file("sub/nested.txt") == "nested content"


def test_dotdot_traversal_rejected(workspace):
    with pytest.raises(ToolExecutionError):
        read_file("../outside/secret.txt")


def test_absolute_path_escape_rejected(workspace):
    outside_file = workspace.parent / "outside" / "secret.txt"
    with pytest.raises(ToolExecutionError):
        read_file(str(outside_file))


def test_missing_file_rejected(workspace):
    with pytest.raises(ToolExecutionError):
        read_file("does_not_exist.txt")


def test_directory_not_file_rejected(workspace):
    with pytest.raises(ToolExecutionError):
        read_file("sub")


def test_oversized_file_rejected(workspace):
    (workspace / "big.txt").write_text("x" * 200_000)
    with pytest.raises(ToolExecutionError):
        read_file("big.txt")


def test_binary_file_rejected(workspace):
    (workspace / "bin.dat").write_bytes(bytes(range(256)) * 4)
    with pytest.raises(ToolExecutionError):
        read_file("bin.dat")


@pytest.fixture
def multiroot(tmp_path, monkeypatch):
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    notes = tmp_path / "notes"
    project_a.mkdir()
    project_b.mkdir()
    notes.mkdir()

    (project_a / "a.txt").write_text("content of a.txt in project A")
    (project_a / "shared_name.txt").write_text("project A's version")
    (project_b / "b.txt").write_text("content of b.txt in project B")
    (project_b / "shared_name.txt").write_text("project B's version")
    standalone = notes / "standalone.md"
    standalone.write_text("standalone note content")
    (notes / "sibling.md").write_text("sibling - should NOT be readable")

    monkeypatch.setenv("BARBAI_TOOLS_ROOTS", f"{project_a},{project_b},{standalone}")
    return {"a": project_a, "b": project_b, "standalone": standalone, "notes": notes}


def test_relative_resolves_against_first_matching_root(multiroot):
    assert read_file("a.txt") == "content of a.txt in project A"


def test_relative_resolves_against_second_root_when_not_in_first(multiroot):
    assert read_file("b.txt") == "content of b.txt in project B"


def test_ambiguous_name_picks_first_configured_root(multiroot):
    assert read_file("shared_name.txt") == "project A's version"


def test_absolute_path_into_second_root(multiroot):
    assert read_file(str(multiroot["b"] / "b.txt")) == "content of b.txt in project B"


def test_standalone_file_root_by_absolute_path(multiroot):
    assert read_file(str(multiroot["standalone"])) == "standalone note content"


def test_sibling_of_file_root_not_exposed(multiroot):
    with pytest.raises(ToolExecutionError):
        read_file(str(multiroot["notes"] / "sibling.md"))


def test_escapes_root_a_but_lands_in_root_b_is_allowed(multiroot):
    """Subtle case: a relative path with '..' that escapes project-a but
    legitimately lands inside project-b (also an allowed root) must still
    be permitted - it isn't actually escaping the allowlist.
    """
    rel = os.path.relpath(multiroot["b"] / "b.txt", start=multiroot["a"])
    assert read_file(rel) == "content of b.txt in project B"
