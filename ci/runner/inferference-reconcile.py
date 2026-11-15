#!/usr/bin/env python3
"""Run the reconcile CLI from a checkout. A thin shim.

The logic lives in `inferference.reconcile_cli`. It is also the console script
`inferference-reconcile`. The launchers call this path with $INFERFERENCE_PYTHON,
so the interpreter must have `inferference` installed (the library venv has an
editable install; the Nix package has the locked env).
"""

from inferference.reconcile_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
