"""Docker HEALTHCHECK: healthy if either the API or the UI answers."""

import sys
import urllib.request


def ok(url: str) -> bool:
    try:
        return urllib.request.urlopen(url, timeout=4).status == 200
    except Exception:
        return False


sys.exit(0 if ok("http://localhost:8000/health") or ok("http://localhost:7860/") else 1)
