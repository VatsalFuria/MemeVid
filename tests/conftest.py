import os

# Ensure a required setting exists before app.main is imported by test modules.
os.environ.setdefault("PEXELS_API_KEY", "test-key")