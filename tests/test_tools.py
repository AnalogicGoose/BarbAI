import os
import shutil

import pytest

from barbai.core.global_memory import load_facts
from barbai.core.tools import (
    GATED_TOOLS,
    ToolExecutionError,
    build_tool_defs,
    list_directory,
    patch_file,
    read_file,
    remember,
    search,
    write_file,
)

requires_ripgrep = pytest.mark.skipif(
    shutil.which("rg") is None, reason="ripgrep ('rg') not installed on this machine"
)


@pytest.fixture
def global_memory_store(tmp_path, monkeypatch):
    monkeypatch.setenv("BARBAI_GLOBAL_MEMORY_PATH", str(tmp_path / "global_memory.json"))


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


def test_write_new_file(workspace):
    write_file("new.txt", "hello world")
    assert (workspace / "new.txt").read_text() == "hello world"


def test_write_overwrites_existing_file(workspace):
    write_file("hello.txt", "replaced")
    assert read_file("hello.txt") == "replaced"


def test_write_nested_existing_dir(workspace):
    write_file("sub/new_nested.txt", "nested content")
    assert (workspace / "sub" / "new_nested.txt").read_text() == "nested content"


def test_write_missing_parent_dir_rejected_without_create_dirs(workspace):
    with pytest.raises(ToolExecutionError):
        write_file("newdir/new.txt", "content")


def test_write_create_dirs_makes_parents(workspace):
    write_file("newdir/sub/new.txt", "content", create_dirs=True)
    assert (workspace / "newdir" / "sub" / "new.txt").read_text() == "content"


def test_write_dotdot_traversal_rejected(workspace):
    with pytest.raises(ToolExecutionError):
        write_file("../outside/evil.txt", "pwned")


def test_write_absolute_path_escape_rejected(workspace):
    outside_file = workspace.parent / "outside" / "evil.txt"
    with pytest.raises(ToolExecutionError):
        write_file(str(outside_file), "pwned")


def test_write_oversized_content_rejected(workspace):
    with pytest.raises(ToolExecutionError):
        write_file("big.txt", "x" * 2_000_000)


def test_write_rejects_directory_target(workspace):
    with pytest.raises(ToolExecutionError):
        write_file("sub", "content")


def test_write_relative_resolves_against_first_directory_root(multiroot):
    write_file("new_in_a.txt", "goes in project A")
    assert (multiroot["a"] / "new_in_a.txt").read_text() == "goes in project A"


def test_write_relative_overwrites_existing_match_by_priority(multiroot):
    write_file("shared_name.txt", "overwritten")
    assert (multiroot["a"] / "shared_name.txt").read_text() == "overwritten"
    assert (multiroot["b"] / "shared_name.txt").read_text() == "project B's version"


def test_remember_is_gated():
    assert "remember" in GATED_TOOLS


def test_remember_stores_a_fact(global_memory_store):
    remember("likes tea")
    facts = load_facts()
    assert len(facts) == 1
    assert facts[0]["text"] == "likes tea"


def test_remember_strips_whitespace(global_memory_store):
    remember("  likes tea  ")
    assert load_facts()[0]["text"] == "likes tea"


def test_remember_rejects_empty_text(global_memory_store):
    with pytest.raises(ToolExecutionError):
        remember("   ")


def test_remember_rejects_oversized_text(global_memory_store):
    with pytest.raises(ToolExecutionError):
        remember("x" * 501)


def test_remember_rejected_when_disabled(global_memory_store, monkeypatch):
    monkeypatch.setenv("BARBAI_GLOBAL_MEMORY", "off")
    with pytest.raises(ToolExecutionError):
        remember("likes tea")
    assert load_facts() == []


def test_build_tool_defs_includes_remember_when_enabled(global_memory_store, monkeypatch):
    monkeypatch.delenv("BARBAI_GLOBAL_MEMORY", raising=False)
    names = [d["function"]["name"] for d in build_tool_defs()]
    assert "remember" in names


def test_build_tool_defs_excludes_remember_when_disabled(global_memory_store, monkeypatch):
    monkeypatch.setenv("BARBAI_GLOBAL_MEMORY", "off")
    names = [d["function"]["name"] for d in build_tool_defs()]
    assert "remember" not in names


@pytest.fixture
def tree_workspace(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "README.md").write_text("readme")
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("def main():\n    return needle_value\n")
    (root / "src" / "lib.py").write_text("class Helper:\n    pass\n")
    (root / "src" / "sub").mkdir()
    (root / "src" / "sub" / "deep.py").write_text("NEEDLE = 1\n")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "junk.js").write_text("should be filtered out")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("git internals")

    monkeypatch.setenv("BARBAI_TOOLS_ROOTS", str(root))
    return root


def test_list_directory_non_recursive(tree_workspace):
    result = list_directory(".")
    entries = result.splitlines()
    assert "README.md" in entries
    assert "src/" in entries
    assert "node_modules/" in entries  # top-level noise dirs still shown non-recursively


def test_list_directory_recursive_filters_noise_dirs(tree_workspace):
    result = list_directory(".", recursive=True)
    assert "src/main.py" in result
    assert "src/sub/deep.py" in result
    assert "node_modules" not in result
    assert ".git" not in result


def test_list_directory_missing_dir_rejected(tree_workspace):
    with pytest.raises(ToolExecutionError):
        list_directory("does_not_exist")


def test_list_directory_rejects_a_file(tree_workspace):
    with pytest.raises(ToolExecutionError):
        list_directory("README.md")


def test_list_directory_outside_allowlist_rejected(tree_workspace):
    with pytest.raises(ToolExecutionError):
        list_directory(str(tree_workspace.parent))


def test_list_directory_empty(tree_workspace):
    (tree_workspace / "empty_dir").mkdir()
    assert list_directory("empty_dir") == "empty_dir is empty"


def test_list_directory_truncates(tree_workspace, monkeypatch):
    monkeypatch.setattr("barbai.core.tools.MAX_LIST_ENTRIES", 2)
    result = list_directory(".", recursive=True)
    assert "truncated" in result


@requires_ripgrep
def test_search_finds_match(tree_workspace):
    result = search("needle_value")
    assert "src/main.py" in result
    assert "needle_value" in result


@requires_ripgrep
def test_search_no_matches(tree_workspace):
    assert search("nothing_matches_this_xyz") == "no matches"


@requires_ripgrep
def test_search_scoped_to_path(tree_workspace):
    result = search("NEEDLE", path="src/sub")
    assert "deep.py" in result
    result_root = search("needle_value", path="src/sub")
    assert result_root == "no matches"


@requires_ripgrep
def test_search_ignore_case(tree_workspace):
    result = search("NEEDLE_VALUE", ignore_case=True)
    assert "main.py" in result


@requires_ripgrep
def test_search_case_sensitive_by_default(tree_workspace):
    assert search("NEEDLE_VALUE") == "no matches"


def test_search_path_outside_allowlist_rejected(tree_workspace):
    with pytest.raises(ToolExecutionError):
        search("anything", path=str(tree_workspace.parent))


@requires_ripgrep
def test_search_respects_gitignore(tree_workspace):
    (tree_workspace / ".gitignore").write_text("ignored.py\n")
    (tree_workspace / "ignored.py").write_text("needle_value")
    result = search("needle_value")
    assert "ignored.py" not in result
    assert "src/main.py" in result


def test_search_missing_ripgrep(tree_workspace, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(ToolExecutionError):
        search("needle_value")


def test_read_file_range(workspace):
    (workspace / "multi.txt").write_text("line1\nline2\nline3\nline4\n")
    assert read_file("multi.txt", start_line=2, end_line=3) == "line2\nline3\n"


def test_read_file_range_from_start(workspace):
    (workspace / "multi.txt").write_text("line1\nline2\nline3\n")
    assert read_file("multi.txt", end_line=2) == "line1\nline2\n"


def test_read_file_range_to_end(workspace):
    (workspace / "multi.txt").write_text("line1\nline2\nline3\n")
    assert read_file("multi.txt", start_line=2) == "line2\nline3\n"


def test_read_file_range_past_end_of_file_rejected(workspace):
    (workspace / "multi.txt").write_text("line1\nline2\n")
    with pytest.raises(ToolExecutionError):
        read_file("multi.txt", start_line=10)


def test_read_file_range_invalid_order_rejected(workspace):
    (workspace / "multi.txt").write_text("line1\nline2\nline3\n")
    with pytest.raises(ToolExecutionError):
        read_file("multi.txt", start_line=3, end_line=1)


def test_read_file_range_start_below_one_rejected(workspace):
    (workspace / "multi.txt").write_text("line1\nline2\n")
    with pytest.raises(ToolExecutionError):
        read_file("multi.txt", start_line=0)


def test_read_file_range_bypasses_whole_file_size_cap(workspace):
    line = "x" * 100 + "\n"  # 101 bytes/line - small on its own, big in bulk
    (workspace / "big_ranged.txt").write_text(line * 2000)  # ~202,000 bytes total
    # whole-file read still rejected...
    with pytest.raises(ToolExecutionError):
        read_file("big_ranged.txt")
    # ...but a narrow range within it works
    assert read_file("big_ranged.txt", start_line=1, end_line=1) == line


def test_read_file_range_oversized_slice_still_rejected(workspace):
    line = "x" * 100 + "\n"
    (workspace / "big_ranged.txt").write_text(line * 2000)  # ~202,000 bytes total
    with pytest.raises(ToolExecutionError):
        read_file("big_ranged.txt", start_line=1, end_line=2000)  # the whole thing, via a range


def test_patch_file_replaces_unique_match(workspace):
    (workspace / "code.py").write_text("def foo():\n    return 1\n")
    patch_file("code.py", "return 1", "return 2")
    assert read_file("code.py") == "def foo():\n    return 2\n"


def test_patch_file_no_match_rejected(workspace):
    (workspace / "code.py").write_text("def foo():\n    return 1\n")
    with pytest.raises(ToolExecutionError):
        patch_file("code.py", "return 99", "return 2")


def test_patch_file_ambiguous_match_rejected_without_replace_all(workspace):
    (workspace / "code.py").write_text("x = 1\ny = 1\n")
    with pytest.raises(ToolExecutionError):
        patch_file("code.py", "= 1", "= 2")


def test_patch_file_replace_all(workspace):
    (workspace / "code.py").write_text("x = 1\ny = 1\n")
    patch_file("code.py", "= 1", "= 2", replace_all=True)
    assert read_file("code.py") == "x = 2\ny = 2\n"


def test_patch_file_identical_strings_rejected(workspace):
    (workspace / "code.py").write_text("return 1\n")
    with pytest.raises(ToolExecutionError):
        patch_file("code.py", "return 1", "return 1")


def test_patch_file_empty_old_string_rejected(workspace):
    (workspace / "code.py").write_text("return 1\n")
    with pytest.raises(ToolExecutionError):
        patch_file("code.py", "", "something")


def test_patch_file_missing_file_rejected(workspace):
    with pytest.raises(ToolExecutionError):
        patch_file("does_not_exist.py", "a", "b")


def test_patch_file_rejects_creating_new_files(workspace):
    with pytest.raises(ToolExecutionError):
        patch_file("brand_new.py", "a", "b")
    assert not (workspace / "brand_new.py").exists()


def test_patch_file_outside_allowlist_rejected(workspace):
    with pytest.raises(ToolExecutionError):
        patch_file(str(workspace.parent / "outside.py"), "a", "b")


def test_patch_file_oversized_file_rejected(workspace):
    (workspace / "big.py").write_text("x" * 200_000)
    with pytest.raises(ToolExecutionError):
        patch_file("big.py", "x", "y")


def test_patch_file_oversized_result_rejected(workspace):
    (workspace / "code.py").write_text("MARKER\n")
    with pytest.raises(ToolExecutionError):
        patch_file("code.py", "MARKER", "y" * 2_000_000)


def test_patch_file_is_gated():
    assert "patch_file" in GATED_TOOLS


# --- Cross-platform: line-ending preservation (Windows CRLF vs Unix LF) ---


def test_patch_file_preserves_crlf_line_endings(workspace):
    raw = b"def foo():\r\n    return 1\r\n    return 2\r\n"
    (workspace / "crlf.py").write_bytes(raw)
    patch_file("crlf.py", "return 1", "return 100")
    result = (workspace / "crlf.py").read_bytes()
    assert result == b"def foo():\r\n    return 100\r\n    return 2\r\n"
    assert b"\n" not in result.replace(b"\r\n", b"")  # no bare \n snuck in anywhere


def test_patch_file_preserves_lf_line_endings(workspace):
    raw = b"def foo():\n    return 1\n    return 2\n"
    (workspace / "lf.py").write_bytes(raw)
    patch_file("lf.py", "return 1", "return 100")
    result = (workspace / "lf.py").read_bytes()
    assert result == b"def foo():\n    return 100\n    return 2\n"
    assert b"\r" not in result


def test_write_file_preserves_crlf_on_existing_file(workspace):
    (workspace / "crlf.txt").write_bytes(b"line1\r\nline2\r\n")
    write_file("crlf.txt", "line1\nline2\nline3")
    result = (workspace / "crlf.txt").read_bytes()
    assert result == b"line1\r\nline2\r\nline3"


def test_write_file_new_file_uses_lf(workspace):
    write_file("brand_new.txt", "line1\nline2")
    result = (workspace / "brand_new.txt").read_bytes()
    assert result == b"line1\nline2"
    assert b"\r" not in result
