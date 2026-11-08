"""Build the active documentation version and update its entry-point redirect."""

import os
from pathlib import Path
import re
import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parent
    version = os.environ.get("DOC_VERSION", "0.3.0")
    if (
        not version
        or version in (".", "..")
        or any(c not in "0123456789.-abcdefghijklmnopqrstuvwxyz" for c in version)
    ):
        raise ValueError("DOC_VERSION must be a version directory name.")
    environment = os.environ.copy()
    environment["DOC_VERSION"] = version
    environment.setdefault("DOC_VERSIONS", f"{version},0.2.7")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-W",
            "--keep-going",
            "-b",
            "html",
            "-d",
            str(root / ".doctrees" / version),
            str(root / "source"),
            str(root / version),
            *sys.argv[1:],
        ],
        env=environment,
        check=True,
    )
    for page in (root / version).rglob("*.html"):
        original = page.read_text(encoding="utf-8")
        cleaned = re.sub(r"(<link\b[^>\n]*>)[ \t]+\n", r"\1\n", original)
        if cleaned != original:
            page.write_text(cleaned, encoding="utf-8")
    (root / "index.html").write_text(
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f'<meta http-equiv="refresh" content="0; url={version}/">\n'
        f'<link rel="canonical" href="{version}/">\n'
        "<title>Thefittest documentation</title>\n</head>\n<body>\n"
        f'<p>Redirecting to <a href="{version}/">Thefittest {version} documentation</a>.</p>\n'
        "</body>\n</html>\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
