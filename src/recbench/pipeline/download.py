"""Dataset download helpers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class DownloadError(Exception):
    pass


def fetch(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "5",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--speed-time",
        "30",
        "--speed-limit",
        "10240",
        "-C",
        "-",
        "-o",
        str(dest),
        url,
    ]
    try:
        subprocess.run(command, check=True, timeout=timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise DownloadError(f"Could not download {url}: {exc}") from exc
    if not dest.exists() or dest.stat().st_size < 32:
        raise DownloadError(f"Download looked empty: {url}")


def kaggle_download(args: list[str], dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("kaggle") is None:
        raise DownloadError(
            "The kaggle CLI is not installed. Put kaggle.json in ~/.kaggle/ and `pip install kaggle`."
        )
    try:
        subprocess.run(
            ["kaggle", *args, "-p", str(dest_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise DownloadError(exc.stderr or exc.stdout or "kaggle download failed") from exc
