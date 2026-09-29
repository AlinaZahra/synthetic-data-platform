import os

# History autosave queues a background job per explicit generate/export. Off by default in tests (they check it explicitly).
os.environ.setdefault("SDP_AUTOSAVE", "0")
