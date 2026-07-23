"""Locate ffmpeg / COLMAP binaries and validate them.

Users often do not have COLMAP on PATH. We probe common install locations and
let the panel persist an override via prefs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class ToolStatus:
    name: str
    path: str
    found: bool
    version: str = ""
    source: str = ""  # "path" | "hint" | "saved" | "missing"


def which(name: str) -> Optional[str]:
    return shutil.which(name)


def _run_version(argv: list[str]) -> str:
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=8)
        out = (r.stdout or r.stderr or "").strip().splitlines()
        return out[0][:120] if out else ""
    except Exception:
        return ""


def _windows_ffmpeg_hints() -> list[str]:
    home = os.path.expanduser("~")
    local = os.environ.get("LOCALAPPDATA", "")
    program = os.environ.get("ProgramFiles", r"C:\Program Files")
    hints = [
        os.path.join(program, "ffmpeg", "bin", "ffmpeg.exe"),
        os.path.join(local, "Microsoft", "WinGet", "Links", "ffmpeg.exe"),
        os.path.join(home, "scoop", "shims", "ffmpeg.exe"),
        os.path.join(home, "tools", "ffmpeg", "bin", "ffmpeg.exe"),
        # winget Gyan.FFmpeg often lands here:
        os.path.join(local, "Microsoft", "WinGet", "Packages"),
    ]
    # Expand WinGet Packages/*/ffmpeg*/bin/ffmpeg.exe
    packages = os.path.join(local, "Microsoft", "WinGet", "Packages")
    if os.path.isdir(packages):
        try:
            for name in os.listdir(packages):
                if "ffmpeg" not in name.lower() and "gyan" not in name.lower():
                    continue
                root = os.path.join(packages, name)
                for dirpath, _dirnames, filenames in os.walk(root):
                    if "ffmpeg.exe" in filenames:
                        hints.append(os.path.join(dirpath, "ffmpeg.exe"))
                        break
        except OSError:
            pass
    return hints


def refresh_process_path() -> None:
    """Pull updated User/Machine PATH into this process (after winget install)."""
    if os.name != "nt":
        return
    try:
        import winreg

        def _read(root, subkey):
            try:
                with winreg.OpenKey(root, subkey) as k:
                    val, _ = winreg.QueryValueEx(k, "Path")
                    return val or ""
            except OSError:
                return ""

        user = _read(winreg.HKEY_CURRENT_USER, r"Environment")
        machine = _read(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        )
        parts = []
        for chunk in (machine, user, os.environ.get("PATH", "")):
            for p in chunk.split(";"):
                p = p.strip()
                if p and p not in parts:
                    parts.append(p)
        # WinGet Links shim dir
        links = os.path.join(
            os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Links"
        )
        if links and os.path.isdir(links) and links not in parts:
            parts.insert(0, links)
        os.environ["PATH"] = ";".join(parts)
    except Exception:
        pass


def _windows_colmap_hints() -> list[str]:
    home = os.path.expanduser("~")
    local = os.environ.get("LOCALAPPDATA", "")
    program = os.environ.get("ProgramFiles", r"C:\Program Files")
    program86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    hints = [
        os.path.join(program, "COLMAP", "colmap.exe"),
        os.path.join(program, "colmap", "COLMAP.bat"),
        os.path.join(program, "colmap", "colmap.exe"),
        os.path.join(program86, "COLMAP", "colmap.exe"),
        os.path.join(local, "Programs", "COLMAP", "colmap.exe"),
        os.path.join(home, "tools", "colmap", "colmap.exe"),
        os.path.join(home, "colmap", "colmap.exe"),
        r"C:\colmap\colmap.exe",
        r"D:\colmap\colmap.exe",
    ]
    packages = os.path.join(local, "Microsoft", "WinGet", "Packages")
    if os.path.isdir(packages):
        try:
            for name in os.listdir(packages):
                if "colmap" not in name.lower():
                    continue
                root = os.path.join(packages, name)
                for dirpath, _dirnames, filenames in os.walk(root):
                    for exe in ("colmap.exe", "COLMAP.bat"):
                        if exe in filenames:
                            hints.append(os.path.join(dirpath, exe))
                            break
        except OSError:
            pass
    return hints


def _unix_colmap_hints() -> list[str]:
    home = os.path.expanduser("~")
    return [
        "/usr/local/bin/colmap",
        "/opt/homebrew/bin/colmap",
        "/usr/bin/colmap",
        os.path.join(home, "bin", "colmap"),
        os.path.join(home, ".local", "bin", "colmap"),
    ]


def _unix_ffmpeg_hints() -> list[str]:
    return [
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
        "/usr/bin/ffmpeg",
    ]


def resolve_tool(
    name: str,
    saved: str = "",
    exe_names: Optional[list[str]] = None,
    hints: Optional[list[str]] = None,
    version_args: Optional[list[str]] = None,
) -> ToolStatus:
    """Prefer saved override, then PATH, then filesystem hints."""
    exe_names = exe_names or [name]
    version_args = version_args or ["-version"]

    candidates: list[tuple[str, str]] = []
    if saved and saved.strip():
        candidates.append((saved.strip(), "saved"))
    for exe in exe_names:
        hit = which(exe)
        if hit:
            candidates.append((hit, "path"))
    for h in hints or []:
        candidates.append((h, "hint"))

    seen = set()
    for path, source in candidates:
        key = os.path.normcase(os.path.abspath(path)) if os.path.isabs(path) else path
        if key in seen:
            continue
        seen.add(key)
        # Allow bare command names from PATH resolution or saved "colmap"
        if os.path.sep in path or (os.name == "nt" and ":" in path):
            if not os.path.isfile(path):
                continue
            runnable = path
        else:
            runnable = which(path) or path
            if source != "path" and not os.path.isfile(runnable) and which(path) is None:
                # bare name without PATH hit
                if source == "saved":
                    # still try later as command name
                    pass
                else:
                    continue

        ver = _run_version([runnable] + version_args)
        # Accept if version worked OR file exists OR path-resolved
        ok = bool(ver) or os.path.isfile(runnable) or which(runnable) is not None
        if ok:
            return ToolStatus(name=name, path=runnable, found=True, version=ver, source=source)

    return ToolStatus(name=name, path=saved or exe_names[0], found=False, version="", source="missing")


def resolve_ffmpeg(saved: str = "") -> ToolStatus:
    hints = _windows_ffmpeg_hints() if os.name == "nt" else _unix_ffmpeg_hints()
    names = ["ffmpeg.exe", "ffmpeg"] if os.name == "nt" else ["ffmpeg"]
    return resolve_tool("ffmpeg", saved=saved, exe_names=names, hints=hints, version_args=["-version"])


def resolve_colmap(saved: str = "") -> ToolStatus:
    hints = _windows_colmap_hints() if os.name == "nt" else _unix_colmap_hints()
    names = ["colmap.exe", "COLMAP.bat", "colmap"] if os.name == "nt" else ["colmap"]
    # COLMAP prints help on -h; -version may not exist on all builds
    status = resolve_tool("colmap", saved=saved, exe_names=names, hints=hints, version_args=["-h"])
    if status.found and not status.version:
        status.version = "colmap (ok)"
    return status


def format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit, div in (("KB", 1024), ("MB", 1024**2), ("GB", 1024**3), ("TB", 1024**4)):
        v = n / div
        if v < 1024 or unit == "TB":
            return f"{v:.1f} {unit}"
    return f"{n} B"
