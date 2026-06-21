"""wc2026-upset-propagation - conditional fair odds for WC 2026 bracket scenarios."""

from __future__ import annotations

import logging as _logging
from pathlib import Path as _Path

__version__ = "0.1.0"


# Auto-load .env at package import time.
#
# Why here: any code path that touches the framework (CLI runs, tests,
# external integrations) starts by importing `upset_propagation.*`, so this
# is the right central place to populate os.environ from the .env file.
# Without this, env vars (FAIRLINE_API_URL, SLACK_WEBHOOK_URL, etc.) would
# need to be manually exported in every shell session - friction we don't
# want.
#
# The search starts from this file's directory and walks up looking for a
# .env file. This is robust to being run from any working directory
# (e.g. `python -m upset_propagation.run` from a subdirectory, or invoked
# from a parent project that imports us as a library).
#
# Silent on missing: if no .env exists, we proceed with whatever's already
# in os.environ. This is the right behavior: a production deployment sets vars via the
# platform's secret manager, not via a .env file.
try:
    from dotenv import load_dotenv as _load_dotenv

    _here = _Path(__file__).resolve()
    for _candidate_dir in [_here.parent, *_here.parents]:
        _env_path = _candidate_dir / ".env"
        if _env_path.exists():
            _load_dotenv(_env_path)
            break
except ImportError:
    # python-dotenv is in our dependencies, so this shouldn't happen in
    # normal installs. If it does (e.g. minimal env without all deps), we
    # log and continue - manual export still works as a fallback.
    _logging.getLogger(__name__).debug(
        "python-dotenv not installed; .env auto-loading disabled. "
        "Set environment variables manually."
    )