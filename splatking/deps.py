"""Install / download missing tools (ffmpeg, COLMAP, vocab tree).

Binaries are never vendored in the GitHub repo. On Windows we try winget;
vocab trees are fetched on demand into the plugin data directory.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

from .paths import resolve_ffmpeg, resolve_colmap, refresh_process_path

# Compact vocab tree (~15MB) — good default; larger trees can be browsed manually.
VOCAB_TREE_URL = "https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin"
VOCAB_TREE_FILENAME = "vocab_tree_flickr100K_words32K.bin"

WINGET_FFMPEG = "Gyan.FFmpeg"
# COLMAP is often missing from winget; try once then fall back to manual URL.
WINGET_COLMAP = "COLMAP.COLMAP"


@dataclass
class InstallResult:
    ok: bool
    message: str
    ffmpeg_path: str = ""
    colmap_path: str = ""
    vocab_path: str = ""


def data_dir() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(here, ".splatking_data")
    os.makedirs(d, exist_ok=True)
    return d


def _winget_available() -> bool:
    return shutil.which("winget") is not None


def _winget_install(package_id: str, on_progress: Optional[Callable[[str], None]] = None) -> tuple[bool, str]:
    if not _winget_available():
        return False, "winget not found — install from https://aka.ms/getwinget or browse to the binary."
    if on_progress:
        on_progress(f"winget install {package_id}...")
    try:
        r = subprocess.run(
            [
                "winget",
                "install",
                "-e",
                "--id",
                package_id,
                "--accept-package-agreements",
                "--accept-source-agreements",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        out = ((r.stdout or "") + "\n" + (r.stderr or "")).strip()
        if r.returncode == 0:
            return True, f"Installed or already present: {package_id}"
        low = out.lower()
        if "already installed" in low or "no available upgrade" in low:
            return True, f"Already installed: {package_id}"
        # Keep message short for UI status line
        return False, f"winget failed for {package_id} (code {r.returncode})"
    except subprocess.TimeoutExpired:
        return False, f"winget timed out installing {package_id}"
    except OSError as e:
        return False, f"winget error: {e}"


def install_missing_tools(
    ffmpeg_saved: str = "",
    colmap_saved: str = "",
    on_progress: Optional[Callable[[str], None]] = None,
) -> InstallResult:
    """Detect ffmpeg/COLMAP; on Windows try winget for anything missing."""
    messages: list[str] = []
    ff = resolve_ffmpeg(ffmpeg_saved)
    cm = resolve_colmap(colmap_saved)

    if os.name != "nt":
        hints = []
        if not ff.found:
            hints.append("ffmpeg: brew install ffmpeg / apt install ffmpeg")
        if not cm.found:
            hints.append("COLMAP: brew install colmap / build from source")
        if hints:
            return InstallResult(
                ok=False,
                message="Auto-install is Windows/winget only. " + " | ".join(hints),
                ffmpeg_path=ff.path if ff.found else "",
                colmap_path=cm.path if cm.found else "",
            )
        return InstallResult(
            ok=True,
            message="ffmpeg and COLMAP already found.",
            ffmpeg_path=ff.path,
            colmap_path=cm.path,
        )

    if not ff.found:
        ok, msg = _winget_install(WINGET_FFMPEG, on_progress)
        messages.append(msg)
        refresh_process_path()
        ff = resolve_ffmpeg("")
        if not ff.found and ok:
            messages.append(
                "ffmpeg installed but not found yet — click Detect/Re-check, "
                "or Browse to WinGet Packages ffmpeg.exe"
            )

    if not cm.found:
        ok, msg = _winget_install(WINGET_COLMAP, on_progress)
        messages.append(msg)
        refresh_process_path()
        cm = resolve_colmap("")
        if not cm.found:
            messages.append(
                "COLMAP is not on winget for this PC. Install from "
                "https://github.com/colmap/colmap/releases then Browse to colmap.exe "
                "(Prepare still writes run_colmap.bat without it)."
            )

    ff = resolve_ffmpeg(ff.path if ff.found else ffmpeg_saved)
    cm = resolve_colmap(cm.path if cm.found else colmap_saved)
    ok = ff.found  # COLMAP optional for prepare scripts
    summary = "; ".join(messages) if messages else "Tools already present."
    if ff.found and cm.found:
        summary = "ffmpeg and COLMAP OK. " + summary
    elif ff.found:
        summary = "ffmpeg OK; COLMAP optional. " + summary
    return InstallResult(
        ok=ok,
        message=summary.strip(),
        ffmpeg_path=ff.path if ff.found else "",
        colmap_path=cm.path if cm.found else "",
    )


def download_vocab_tree(
    on_progress: Optional[Callable[[str], None]] = None,
    url: str = VOCAB_TREE_URL,
    filename: str = VOCAB_TREE_FILENAME,
) -> InstallResult:
    """Download COLMAP vocab tree into plugin data dir if missing."""
    dest = os.path.join(data_dir(), filename)
    if os.path.isfile(dest) and os.path.getsize(dest) > 1_000_000:
        return InstallResult(ok=True, message=f"Vocab tree already present: {dest}", vocab_path=dest)

    if on_progress:
        on_progress(f"Downloading vocab tree (~15MB) to {dest}...")
    tmp = dest + ".part"
    try:
        def _reporthook(block_num, block_size, total_size):
            if not on_progress or total_size <= 0:
                return
            done = block_num * block_size
            pct = min(100, int(100 * done / total_size))
            if block_num % 200 == 0:
                on_progress(f"Vocab tree download: {pct}%")

        urllib.request.urlretrieve(url, tmp, reporthook=_reporthook)
        os.replace(tmp, dest)
    except Exception as e:
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return InstallResult(
            ok=False,
            message=(
                f"Vocab download failed: {e}. "
                f"Browse manually to a .bin from {url}"
            ),
        )
    if on_progress:
        on_progress(f"Vocab tree ready: {dest}")
    return InstallResult(ok=True, message=f"Downloaded vocab tree: {dest}", vocab_path=dest)
