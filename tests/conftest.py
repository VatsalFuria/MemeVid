import os

from dotenv import load_dotenv

# Load real values from .env (e.g. a genuine PEXELS_API_KEY for the
# `network`-marked live tests) before falling back to a placeholder below.
# override=False (the default) means an already-exported env var or a
# real .env value always wins over "test-key".
load_dotenv()

# Ensure a required setting exists even with no .env at all, for tests that
# don't care about a real value (e.g. test_health.py).
os.environ.setdefault("PEXELS_API_KEY", "test-key")