"""Per-wrapper import isolation for the vendored third-party repos.

The repos under ``vismatch/third_party`` were never designed to share one interpreter: many claim
the same top-level module names (``src``, ``third_party``, ``tools``, ``model``, ``models``,
``utils``, ``modules``, ``lightglue``, ...), some shadow installed packages (MatchAnything vendors
a ``roma`` fork that hides the pip ``roma`` rotation library used by dust3r), and several import
these names lazily at construction or inference time, long after their wrapper module was first
imported (MINIMA's ``load_*`` helpers, EDM's yacs-executed configs, xfeat's lighterglue). With a
single flat ``sys.modules``/``sys.path``, whichever repo imports a contested name first owns it for
the whole process, and PEP 420 makes it unrecoverable: a regular package anywhere on ``sys.path``
beats a same-named namespace directory regardless of path order, so no amount of path reordering or
module flushing helps (issue #60).

``ImportSandbox`` gives each ``vismatch.im_models.<wrapper>`` module its own import universe. While
a wrapper's code runs, only that wrapper's third-party directories are on ``sys.path`` and only its
third-party modules are in ``sys.modules``; everything else is parked aside and restored when the
sandbox exits. Module objects are stashed, never destroyed, so live matchers keep working across
swaps. Outside any sandbox the interpreter holds no third-party modules or paths at all, which also
keeps pip packages visible to user code.

Sandboxes are entered automatically at the only three points where third-party code runs:

1. wrapper module import -- ``SandboxedWrapperFinder``, registered in ``vismatch.im_models``;
2. matcher ``__init__`` -- wrapped by ``BaseMatcher.__init_subclass__``;
3. matcher ``_forward`` -- same, covering inference-time lazy imports and direct inner
   ``_forward`` calls (EnsembleMatcher, Keypt2SubpxMatcher).

Entries nest LIFO, so wrappers that build other wrappers work without special cases. Windows from
different threads are serialized by a process-wide lock; threads outside vismatch that import
contested top-level names exactly while a window is open can still mis-resolve them, so avoid
import-heavy background threads during matcher calls.
"""

import functools
import importlib.abc
import importlib.machinery
import os
import sys
import threading
from pathlib import Path

_THIRD_PARTY_DIR = str(Path(__file__).resolve().parent / "third_party")

# Extra directories whose modules should be treated as sandbox-owned (used by the unit tests;
# could also serve out-of-tree vendored repos). Once seen, a prefix is never forgotten: modules
# already pocketed under an old prefix (e.g. after the user changes TORCH_HOME mid-process) must
# keep being recognized, otherwise they would leak into the global interpreter on a later swap.
_EXTRA_PREFIXES: list = []
_SEEN_PREFIXES: set = set()


def _isolation_prefixes() -> tuple[str, ...]:
    """Directory prefixes whose modules belong to sandboxes: the vendored repos and the hub cache.

    The torch hub cache is included because ``torch.hub.load`` executes a repo's hubconf with the
    checkout dir on ``sys.path``, leaving its top-level modules cached afterwards (e.g. the
    keypt2subpx checkout has a top-level ``model.py``, the accelerated_features one a ``modules``
    package). The returned set only ever grows (see ``_SEEN_PREFIXES``).
    """
    _SEEN_PREFIXES.add(_THIRD_PARTY_DIR.rstrip(os.sep) + os.sep)
    try:
        import torch.hub

        _SEEN_PREFIXES.add(torch.hub.get_dir().rstrip(os.sep) + os.sep)
    except Exception:
        pass
    for extra in _EXTRA_PREFIXES:
        _SEEN_PREFIXES.add(str(extra).rstrip(os.sep) + os.sep)
    return tuple(_SEEN_PREFIXES)


def _is_sandboxed_module(module, prefixes: tuple[str, ...]) -> bool:
    """Whether *module* was loaded from a sandboxed location."""
    try:
        origin = getattr(module, "__file__", None)
    except Exception:
        origin = None
    if origin:
        return origin.startswith(prefixes)
    try:
        path = getattr(module, "__path__", None)
        portions = list(path) if path is not None else []
    except KeyError:
        # _NamespacePath._recalculate raises KeyError once the parent package has been parked out
        # of sys.modules; only third-party parents are ever parked, so this is third-party.
        return True
    except Exception:
        return False
    return any(str(portion).startswith(prefixes) for portion in portions)


def _module_family(name: str) -> list[str]:
    """All sys.modules keys belonging to *name*: the module itself and its submodules."""
    prefix = name + "."
    return [key for key in sys.modules if key == name or key.startswith(prefix)]


def _importable_names(paths) -> set[str]:
    """Top-level names that *paths* provide as regular packages or modules.

    Used to decide which cached user/pip modules to shelve while a sandbox is active. Namespace
    dirs (no __init__.py) are deliberately excluded: under PEP 420 a regular package or module
    anywhere on sys.path beats a namespace dir no matter the order, so shelving the user's module
    could never make the repo's namespace dir win -- it would only force a pointless re-import
    (re-running the user's module side effects). Regular children do win by path order, so for
    them shelving is both necessary and sufficient.
    """
    names = set()
    for path in paths:
        base = Path(path)
        if not base.is_dir():
            continue  # tolerate junk entries appended by third-party code (e.g. MINIMA's cwd-relative one)
        try:
            children = list(base.iterdir())
        except OSError:
            continue
        for child in children:
            if child.is_dir():
                if child.name.isidentifier() and (child / "__init__.py").is_file():
                    names.add(child.name)
            elif child.name.endswith(".py") and child.name != "__init__.py":
                stem = child.name[:-3]
                if stem.isidentifier():
                    names.add(stem)
    return names


class ImportSandbox:
    """Context manager that virtualizes ``sys.path`` and ``sys.modules`` for one wrapper module.

    A process-wide re-entrant lock serializes the swap windows: concurrent matcher calls from
    different threads queue up instead of corrupting the shared ``sys.modules``/``sys.path``
    state. The acquire carries a generous timeout so that a pathological cross-thread wait (e.g.
    two threads importing wrapper modules that import each other) surfaces as a clear error
    instead of a silent deadlock.
    """

    _instances: dict = {}
    _stack: list = []  # [(sandbox, frame)], LIFO; frame is None for re-entrant no-op entries
    _lock = threading.RLock()
    _lock_timeout_s = 600
    _owner_thread = None  # ident of the thread currently holding the swap windows

    def __init__(self, key: str):
        self.key = key
        self.paths: list[str] = []  # registered third-party dirs, in add_to_path() order
        self.modules: dict = {}  # parked {name: module} while the sandbox is inactive

    @classmethod
    def get(cls, key: str) -> "ImportSandbox":
        sandbox = cls._instances.get(key)
        if sandbox is None:
            sandbox = cls._instances[key] = cls(key)
        return sandbox

    @classmethod
    def active(cls) -> "ImportSandbox | None":
        """The innermost sandbox active in the *current thread*, or None.

        Other threads must not see (or register paths into) a window they do not own.
        """
        if not cls._stack or cls._owner_thread != threading.get_ident():
            return None
        return cls._stack[-1][0]

    def add_path(self, path: str) -> None:
        """Register *path* with this sandbox and put it in front of ``sys.path``.

        Called by ``vismatch.utils.add_to_path`` while this sandbox is active.
        """
        if path not in self.paths:
            self.paths.append(path)
        # A sandbox owns whatever it registers: counting the dir as an isolation prefix means
        # repos outside vismatch/third_party (out-of-tree matchers) are isolated all the same.
        _SEEN_PREFIXES.add(path.rstrip(os.sep) + os.sep)
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
        frame = self._current_frame()
        if frame is not None:
            prefixes = _isolation_prefixes()
            for name in _importable_names([path]):
                self._displace(name, frame, prefixes)

    def _current_frame(self) -> dict | None:
        for sandbox, frame in reversed(ImportSandbox._stack):
            if sandbox is self and frame is not None:
                return frame
        return None

    @staticmethod
    def _displace(name: str, frame: dict, prefixes: tuple[str, ...]) -> None:
        """Shelve a non-sandboxed module family that this sandbox shadows (e.g. pip ``roma``)."""
        module = sys.modules.get(name)
        if module is None or _is_sandboxed_module(module, prefixes):
            return
        for key in _module_family(name):
            frame["displaced"].setdefault(key, sys.modules.pop(key))

    def __enter__(self) -> "ImportSandbox":
        if not ImportSandbox._lock.acquire(timeout=ImportSandbox._lock_timeout_s):
            raise RuntimeError(
                "Timed out waiting for another thread's vismatch matcher call to finish. "
                "Matcher construction and inference swap global import state and are serialized "
                "across threads; run vismatch calls from one thread at a time."
            )
        try:
            ImportSandbox._owner_thread = threading.get_ident()
            if ImportSandbox.active() is self:
                ImportSandbox._stack.append((self, None))  # re-entrant: our world is already installed
                return self
            prefixes = _isolation_prefixes()
            frame = {"parked": {}, "displaced": {}, "removed_paths": [], "path_snapshot": []}
            # Park every sandboxed module currently visible: they belong to the suspended sandbox
            # (when nesting) or are pre-sandbox leftovers; either way they must not resolve in here.
            for name, module in list(sys.modules.items()):
                if _is_sandboxed_module(module, prefixes):
                    frame["parked"][name] = module
                    del sys.modules[name]
            # Same for sandboxed sys.path entries.
            for entry in list(sys.path):
                if (entry.rstrip(os.sep) + os.sep).startswith(prefixes):
                    frame["removed_paths"].append(entry)
                    sys.path.remove(entry)
            # Install this sandbox's dirs in registration order, each at the front, so the most
            # recently registered one wins, mirroring the original sequence of add_to_path() calls.
            for entry in self.paths:
                if entry in sys.path:
                    sys.path.remove(entry)
                sys.path.insert(0, entry)
            # Shelve non-sandboxed modules that our dirs shadow (pip ``roma`` vs MatchAnything's
            # fork) or that collide with names about to be installed from the pocket.
            for name in _importable_names(self.paths) | {key.split(".", 1)[0] for key in self.modules}:
                self._displace(name, frame, prefixes)
            sys.modules.update(self.modules)
            self.modules = {}
            frame["path_snapshot"] = list(sys.path)
            ImportSandbox._stack.append((self, frame))
            return self
        except BaseException:
            if not ImportSandbox._stack:
                ImportSandbox._owner_thread = None
            ImportSandbox._lock.release()
            raise

    def __exit__(self, exc_type, exc_value, exc_traceback) -> bool:
        try:
            _sandbox, frame = ImportSandbox._stack.pop()
            if frame is None:
                return False
            prefixes = _isolation_prefixes()
            # Pocket everything sandboxed: whatever ran in here imported it, or __enter__ installed it.
            for name, module in list(sys.modules.items()):
                if _is_sandboxed_module(module, prefixes):
                    self.modules[name] = module
                    del sys.modules[name]
            # Third-party code may add its own sys.path entries at runtime (immatch appends the r2d2
            # dir, MINIMA a relative RoMa dir): adopt them, then take all our entries off the path.
            for entry in list(sys.path):
                if entry not in frame["path_snapshot"] and entry not in self.paths:
                    self.paths.append(entry)
            for entry in self.paths:
                while entry in sys.path:
                    sys.path.remove(entry)
            # Restore the outside world exactly as __enter__ found it.
            for entry in reversed(frame["removed_paths"]):
                sys.path.insert(0, entry)
            sys.modules.update(frame["displaced"])
            sys.modules.update(frame["parked"])
            return False
        finally:
            if not ImportSandbox._stack:
                ImportSandbox._owner_thread = None
            ImportSandbox._lock.release()


def sandboxed_method(func, key: str):
    """Wrap *func* so it runs inside ``ImportSandbox.get(key)``. Idempotent."""
    if getattr(func, "_sandbox_key", None) is not None:
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with ImportSandbox.get(key):
            return func(*args, **kwargs)

    wrapper._sandbox_key = key
    return wrapper


class _SandboxedLoader:
    """Delegating loader that executes the wrapped module inside its sandbox."""

    def __init__(self, loader, key: str):
        self._loader = loader
        self._key = key

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        with ImportSandbox.get(self._key):
            self._loader.exec_module(module)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class SandboxedWrapperFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder that imports each ``<package>.<wrapper>`` module inside its own sandbox."""

    def __init__(self, package: str):
        self._prefix = package + "."
        self._depth = package.count(".") + 1

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self._prefix) or fullname.count(".") != self._depth:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _SandboxedLoader(spec.loader, fullname)
        return spec
