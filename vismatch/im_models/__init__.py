"""Wrapper modules for the supported matchers.

Importing ``vismatch.im_models.<wrapper>`` executes the wrapper inside its own ImportSandbox, so
the third-party imports it performs (and the sys.path dirs it registers via add_to_path) stay
private to that wrapper instead of leaking into the global interpreter state. See
vismatch/import_sandbox.py for the full rationale.
"""

import sys

from vismatch.import_sandbox import SandboxedWrapperFinder

if not any(isinstance(finder, SandboxedWrapperFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, SandboxedWrapperFinder(__name__))
