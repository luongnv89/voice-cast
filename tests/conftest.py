"""Shared pytest configuration."""

import importlib
import os
import sys
import types
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Heavyweight optional dependencies that the suite must never require (CI
# installs none of them). A meta-path finder serves lightweight stand-ins only
# when the real package is absent, so every test file -- run alone or together
# -- sees identical import behavior with no cross-file mock leakage.
_STUB_MODULES = frozenset({"torch", "transformers", "TTS", "chatterbox", "mlx_audio"})


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return MagicMock(name=name)


class _StubLoader(Loader):
    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        """Stubs are fully formed at creation; there is nothing to execute."""


class _OptionalDependencyFinder(MetaPathFinder):
    """Resolve allowlisted packages to stubs only when they are not installed."""

    _resolving = False

    def find_spec(self, fullname, path=None, target=None):
        if self._resolving or fullname not in _STUB_MODULES:
            return None
        type(self)._resolving = True
        try:
            importlib.util.find_spec(fullname)
        except ImportError:
            return spec_from_loader(fullname, _StubLoader())
        else:
            return None
        finally:
            type(self)._resolving = False


sys.meta_path.insert(0, _OptionalDependencyFinder())
