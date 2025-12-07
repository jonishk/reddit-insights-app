# reset_tracker.py
"""
Attempt to remove common tracker DB files so your tracker can be re-initialized.
"""

import os
from pathlib import Path

candidates = [
    "data/tracker.db", "data/indexer.db", "data/pipeline.db",
]

deleted = []
for p in candidates:
    fp = Path(p)
    if fp.exists():
        try:
            fp.unlink()
            deleted.append(str(fp))
        except Exception as e:
            print(f"Could not delete {fp}: {e}")

if deleted:
    print("Deleted DB files:", deleted)
else:
    print("No tracker DB files found among candidates.")
