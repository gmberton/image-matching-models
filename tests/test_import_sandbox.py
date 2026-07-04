"""Unit tests for vismatch.import_sandbox -- no model weights, network, or GPU needed.

These check the machinery's contract directly, with synthetic repos under tmp_path registered as
sandbox-owned via ``_EXTRA_PREFIXES``:

- while a sandbox is active, only its modules/paths are visible; same-named modules from other
  sandboxes are parked with object identity preserved;
- outside any sandbox the interpreter holds no sandboxed modules or paths;
- a user/pip module shadowed by a sandbox's regular package is shelved during the window and
  restored with identity intact, and is never re-executed;
- namespace-only sandbox dirs do NOT shelve user modules (PEP 420 makes that futile);
- windows nest LIFO and are re-entrant; exceptions inside a window restore everything;
- sys.path entries appended by code running inside a window are adopted by that sandbox;
- concurrent windows from different threads are serialized, not interleaved.
"""

import importlib
import sys
import threading
import time

import pytest

from vismatch import import_sandbox
from vismatch.import_sandbox import ImportSandbox

PKG = "vmtest_pkg"


def make_provider(root, name, mark, regular=True):
    """Create a repo dir at *root* providing package *name* with the given MARK."""
    pkg_dir = root / name
    pkg_dir.mkdir(parents=True)
    code = f"open(r'{root / 'exec_counts.txt'}', 'a').write('{mark}\\n')\nMARK = '{mark}'\n"
    if regular:
        (pkg_dir / "__init__.py").write_text(code)
    else:
        (pkg_dir / "leaf.py").write_text(code)
    return root


def exec_count(root, mark):
    counts = root / "exec_counts.txt"
    return counts.read_text().split().count(mark) if counts.exists() else 0


def sandbox_for(tmp_path, label, repo):
    """Fresh, uniquely-keyed sandbox owning *repo* (keys are global, tmp_path.name is unique)."""
    sandbox = ImportSandbox.get(f"vmtest.{tmp_path.name}.{label}")
    sandbox.paths.append(str(repo))
    return sandbox


@pytest.fixture(autouse=True)
def isolated_world(tmp_path):
    """Register tmp_path as sandbox-owned territory and scrub global state afterwards."""
    import_sandbox._EXTRA_PREFIXES.append(str(tmp_path))
    yield
    assert not ImportSandbox._stack, "a test left a sandbox window open"
    for name in [n for n in sys.modules if n.startswith("vmtest")]:
        del sys.modules[name]
    sys.path[:] = [p for p in sys.path if not p.startswith(str(tmp_path))]
    import_sandbox._EXTRA_PREFIXES.remove(str(tmp_path))


def assert_outside_clean(tmp_path):
    leaked = [n for n, m in sys.modules.items() if (getattr(m, "__file__", "") or "").startswith(str(tmp_path))]
    assert not leaked, f"sandboxed modules leaked into global sys.modules: {leaked}"
    assert not [p for p in sys.path if p.startswith(str(tmp_path))], "sandboxed paths leaked into sys.path"


def test_isolation_and_identity_across_sandboxes(tmp_path):
    sandbox_a = sandbox_for(tmp_path, "a", make_provider(tmp_path / "repo_a", PKG, "A"))
    sandbox_b = sandbox_for(tmp_path, "b", make_provider(tmp_path / "repo_b", PKG, "B"))

    with sandbox_a:
        mod_a = importlib.import_module(PKG)
        assert mod_a.MARK == "A"
    assert_outside_clean(tmp_path)

    with sandbox_b:
        assert importlib.import_module(PKG).MARK == "B"
    assert_outside_clean(tmp_path)

    with sandbox_a:  # parked module comes back as the same object, not a re-import
        assert sys.modules[PKG] is mod_a
        assert sys.modules[PKG].MARK == "A"
    assert exec_count(tmp_path / "repo_a", "A") == 1
    assert exec_count(tmp_path / "repo_b", "B") == 1


def test_user_module_shelved_and_restored_with_identity(tmp_path):
    user_root = make_provider(tmp_path / "user", PKG, "user")
    import_sandbox._EXTRA_PREFIXES.remove(str(tmp_path))  # user dir must NOT count as sandbox-owned
    import_sandbox._EXTRA_PREFIXES.append(str(tmp_path / "repo_a"))
    sys.path.insert(0, str(user_root))
    user_mod = importlib.import_module(PKG)
    assert user_mod.MARK == "user"

    sandbox_a = sandbox_for(tmp_path, "a", make_provider(tmp_path / "repo_a", PKG, "A"))
    with sandbox_a:
        assert importlib.import_module(PKG).MARK == "A"  # sandbox's regular package wins inside
    assert sys.modules[PKG] is user_mod  # user module restored, same object
    assert importlib.import_module(PKG).MARK == "user"
    assert exec_count(tmp_path / "user", "user") == 1  # user code never re-executed
    import_sandbox._EXTRA_PREFIXES.append(str(tmp_path))  # let the fixture teardown run unchanged


def test_namespace_dir_does_not_shelve_user_module(tmp_path):
    user_root = make_provider(tmp_path / "user", PKG, "user")
    import_sandbox._EXTRA_PREFIXES.remove(str(tmp_path))
    import_sandbox._EXTRA_PREFIXES.append(str(tmp_path / "repo_ns"))
    sys.path.insert(0, str(user_root))
    user_mod = importlib.import_module(PKG)

    sandbox_ns = sandbox_for(tmp_path, "ns", make_provider(tmp_path / "repo_ns", PKG, "NS", regular=False))
    with sandbox_ns:
        # the sandbox only offers a namespace dir: shelving the user's regular module could never
        # make the namespace win (PEP 420), so the user's module must stay untouched
        assert sys.modules[PKG] is user_mod
    assert sys.modules[PKG] is user_mod
    assert exec_count(tmp_path / "user", "user") == 1
    import_sandbox._EXTRA_PREFIXES.append(str(tmp_path))


def test_nesting_is_lifo(tmp_path):
    sandbox_a = sandbox_for(tmp_path, "a", make_provider(tmp_path / "repo_a", PKG, "A"))
    sandbox_b = sandbox_for(tmp_path, "b", make_provider(tmp_path / "repo_b", PKG, "B"))
    with sandbox_a:
        mod_a = importlib.import_module(PKG)
        with sandbox_b:
            assert importlib.import_module(PKG).MARK == "B"
        assert sys.modules[PKG] is mod_a  # A's world restored after the nested window
    assert_outside_clean(tmp_path)


def test_reentry_is_noop(tmp_path):
    sandbox_a = sandbox_for(tmp_path, "a", make_provider(tmp_path / "repo_a", PKG, "A"))
    with sandbox_a:
        mod_a = importlib.import_module(PKG)
        with sandbox_a:
            assert sys.modules[PKG] is mod_a
        assert sys.modules[PKG] is mod_a
    assert_outside_clean(tmp_path)


def test_runtime_path_addition_is_adopted(tmp_path):
    extra = tmp_path / "repo_a" / "extra_dir"
    extra.mkdir(parents=True)
    sandbox_a = sandbox_for(tmp_path, "a", make_provider(tmp_path / "repo_a", PKG, "A"))
    with sandbox_a:
        sys.path.append(str(extra))  # what immatch does at runtime
    assert str(extra) not in sys.path
    assert str(extra) in sandbox_a.paths
    with sandbox_a:
        assert str(extra) in sys.path
    assert_outside_clean(tmp_path)


def test_out_of_tree_repo_registered_via_add_to_path(tmp_path):
    """A sandbox owns whatever add_to_path registers, even outside vismatch/third_party."""
    from vismatch.utils import add_to_path

    import_sandbox._EXTRA_PREFIXES.remove(str(tmp_path))  # ownership must come from add_to_path alone
    repo = make_provider(tmp_path / "oot_repo", PKG, "OOT")
    sandbox = ImportSandbox.get(f"vmtest.{tmp_path.name}.oot")
    with sandbox:
        add_to_path(repo)
        assert importlib.import_module(PKG).MARK == "OOT"
    assert PKG not in sys.modules
    assert_outside_clean(tmp_path)
    with sandbox:
        assert sys.modules[PKG].MARK == "OOT"  # pocketed on exit, reinstalled on re-entry
    import_sandbox._EXTRA_PREFIXES.append(str(tmp_path))


def test_exception_inside_window_restores_everything(tmp_path):
    sandbox_a = sandbox_for(tmp_path, "a", make_provider(tmp_path / "repo_a", PKG, "A"))
    path_before = list(sys.path)
    with pytest.raises(RuntimeError, match="boom"):
        with sandbox_a:
            importlib.import_module(PKG)
            raise RuntimeError("boom")
    assert list(sys.path) == path_before
    assert_outside_clean(tmp_path)
    with sandbox_a:  # and the sandbox is still usable afterwards
        assert importlib.import_module(PKG).MARK == "A"


def test_concurrent_windows_are_serialized(tmp_path):
    sandbox_a = sandbox_for(tmp_path, "a", make_provider(tmp_path / "repo_a", PKG, "A"))
    sandbox_b = sandbox_for(tmp_path, "b", make_provider(tmp_path / "repo_b", PKG, "B"))
    errors = []

    def worker(sandbox, mark):
        try:
            for _ in range(10):
                with sandbox:
                    mod = importlib.import_module(PKG)
                    time.sleep(0.002)
                    assert sys.modules[PKG] is mod and mod.MARK == mark
        except Exception as exc:  # pragma: no cover - only on regression
            errors.append(f"{mark}: {exc!r}")

    threads = [
        threading.Thread(target=worker, args=(sandbox_a, "A")),
        threading.Thread(target=worker, args=(sandbox_b, "B")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert_outside_clean(tmp_path)
