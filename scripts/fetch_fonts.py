"""Optional: download Noto fonts into backend/sdp/fonts so PDFs embed Noto instead of a system fallback.

Nothing is downloaded unless you pass --yes. Run it yourself; it makes network requests to github.com.

    python scripts/fetch_fonts.py          # lists what would be fetched
    python scripts/fetch_fonts.py --yes    # downloads (about 1-2 MB in total)

Noto is licensed under the SIL Open Font License 1.1 (free to embed). The URLs below follow the notofonts.github.io
repository layout; if a file moves, the script reports the HTTP error and continues. In Docker, `fonts-noto-core` (apt) is
used instead, so this script is only for local development.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://github.com/notofonts/notofonts.github.io/raw/main/fonts"
FILES = {
    "NotoNaskhArabic-Regular.ttf": f"{BASE}/NotoNaskhArabic/hinted/ttf/NotoNaskhArabic-Regular.ttf",
    "NotoNaskhArabic-Bold.ttf": f"{BASE}/NotoNaskhArabic/hinted/ttf/NotoNaskhArabic-Bold.ttf",
    "NotoSansDevanagari-Regular.ttf": f"{BASE}/NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-Regular.ttf",
    "NotoSans-Regular.ttf": f"{BASE}/NotoSans/hinted/ttf/NotoSans-Regular.ttf",
    "NotoSans-Bold.ttf": f"{BASE}/NotoSans/hinted/ttf/NotoSans-Bold.ttf",
}
DEST = Path(__file__).resolve().parent.parent / "backend" / "sdp" / "fonts"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="actually download")
    args = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        target = DEST / name
        if target.exists():
            print(f"have    {name}")
            continue
        if not args.yes:
            print(f"would fetch {name}\n        from {url}")
            continue
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                target.write_bytes(r.read())
            print(f"fetched {name} ({target.stat().st_size // 1024} KB)")
        except (urllib.error.URLError, OSError) as e:
            print(f"FAILED  {name}: {e}", file=sys.stderr)
    if not args.yes:
        print("\nDry run only. Re-run with --yes to download.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
