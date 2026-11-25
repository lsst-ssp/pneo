"""Where the tests find the digest2 population model.

The tests use the same model file an installed pneo does: ``$PNEO_MODEL`` if
set, else the per-user cache that ``pneo fetch-model`` writes to.  The path is
resolved once, when this module is imported, which is before the autouse
fixture in ``conftest.py`` points ``PNEO_CACHE_DIR`` at an empty directory for
each test.  Tests that need the model skip when it is absent.
"""

import os

from pneo.paths import default_model_path

MODEL_PATH = os.environ.get("PNEO_MODEL", "").strip() or str(default_model_path())
